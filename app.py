import os
import time
from typing import Optional, Any
from pathlib import Path
from operator import itemgetter

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

DATA_PATH = "./data"
LLM_MODEL_NAME = "qwen3"
#EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
# 使用一個支援多語言的 HuggingFace 模型，以便處理中文和英文
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

VECTOR_DB_PATH = "FAISS_DB"

# 聊天記錄儲存 (簡易的記憶體內儲存)
store = {}

Langfuse(host = "http://localhost:3000",
        secret_key = "YOUR_KEY",
        public_key = "YOUR_KEY"
        )

langfuse = get_client()
print("langfuse.auth_check = " + str(langfuse.auth_check()))

langfuse_callback = CallbackHandler()

def get_vectorstore_retriever() -> Optional[Any]:
    """
    載入 PDF、切割文件、建立嵌入並存入 FAISS。
    同時建立 BM25 檢索器。
    返回一個 EnsembleRetriever (混合搜尋)。
    """
    data_dir = Path(DATA_PATH)
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"錯誤：資料夾 '{DATA_PATH}' 不存在。")
        print("請建立 'data' 資料夾並在其中放入 PDF 檔案。")
        return None

    print(f"正在從 '{DATA_PATH}' 載入 PDF 檔案...")
    loader = DirectoryLoader(
        str(data_dir),
        glob = "**/*.pdf",
        loader_cls = PyPDFLoader,
        show_progress = True,
        use_multithreading = True
    )
    
    try:
        docs = loader.load()
        if not docs:
            print(f"在 '{DATA_PATH}' 中找不到 PDF 檔案。")
            return None
        print(f"成功載入 {len(docs)} 個文件頁面。")
    except Exception as e:
        print(f"載入文件時發生錯誤：{e}")
        return None

    # 切割文件
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    splits = text_splitter.split_documents(docs)
    print(f"文件被切割成 {len(splits)} 個區塊。")

    # 建立嵌入模型 (for FAISS)
    print(f"正在初始化嵌入模型：'{EMBED_MODEL_NAME}'...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name = EMBED_MODEL_NAME)
    except Exception as e:
        print(f"初始化 HuggingFace 嵌入模型失敗：{e}")
        print("請確保已安裝 'sentence-transformers'。")
        return None

    # --- 建立兩個獨立的檢索器 ---

    # 建立 FAISS 向量資料庫 (相似性搜尋)
    print("正在建立 FAISS 向量資料庫 (for Vector Search)...")
    try:
        vectorstore = FAISS.from_documents(splits, embeddings)
        # 設定 k=3, 讓它檢索 3 份文件
        faiss_retriever = vectorstore.as_retriever(search_kwargs = {'k' : 3}) 
    except Exception as e:
        print(f"建立 FAISS 索引時出錯：{e}")
        return None
        
    # 建立 BM25 檢索器 (關鍵字搜尋)
    print("正在建立 BM25 檢索器 (for Keyword Search)...")
    try:
        # BM25Retriever 需要原始文件區塊 (splits)
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 3 # 同樣檢索 3 份
    except Exception as e:
        print(f"建立 BM25 索引時出錯：{e}")
        print("請確保已安裝 'rank-bm25' (pip install rank-bm25)")
        return None

    # 建立 Ensemble Retriever (混合搜尋)
    print("正在組合 Ensemble Retriever (混合搜尋)...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.6, 0.4] # [向量權重, 關鍵字權重]
    )
    
    print("RAG 混合檢索器已準備就緒。")
    return ensemble_retriever

# --- 歷史對話管理 ---
HISTORY_LENGTH = -1
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    根據 session_id 取得或建立一個聊天記錄。
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    history: ChatMessageHistory = store[session_id]
    if HISTORY_LENGTH != -1 and len(history.messages) > HISTORY_LENGTH:
        history.messages = history.messages[ -HISTORY_LENGTH: ]

    return history

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    
def create_rag_chain(llm, retriever):
    
    """
    建立核心的 RAG 鏈。
    這個鏈會被 RunnableWithMessageHistory 包裝。
    
    輸入: {"question": str, "history": List[BaseMessage]}
    輸出: {"answer": str, "sources": List[Document]}
    """

    # 檢索鏈 (Retrieval Chain)
    # 這個鏈會接收 {"question": ..., "history": ...}
    # 並輸出 {"sources": [Docs], "question": ..., "history": ...}
    # 使用 itemgetter("question") 來指定只用問題來檢索
    retrieval_docs_chain = RunnableParallel(
        sources = itemgetter("question") | retriever,
        question = itemgetter("question"),
        history = itemgetter("history")
    )
    #print("retrieval_docs_chain = " + str(retrieval_docs_chain) + "\n")

    # 內容格式化鏈 (Context Formatting Chain)
    # 接收上一步的輸出
    # 輸出 {"context": str, "question": ..., "history": ..., "sources": [Docs]}
    format_docs_chain = RunnableParallel(
        context = lambda x : format_docs(x["sources"]),
        question = itemgetter("question"),
        history = itemgetter("history"),
        sources = itemgetter("sources") # 將 sources 透傳下去
    )
    #print("format_docs_chain = " + str(format_docs_chain) + "\n")

    # 提示 (Prompt) + LLM 鏈

    # 定義提示模板
    system_prompt = (
        "你是一個專業的問答助理。"
        "請使用以下提供的'上下文'和'對話紀錄'來回答問題。"
        "如果上下文中沒有相關資訊，請明確告知你不知道，不要編造答案。"
        "請保持答案簡潔。\n\n"
        "上下文:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name = "history"),
        ("human", "{question}"),
    ])
    #print("prompt = " + str(prompt) + "\n")
    
    # RAG 核心鏈
    # 接收 {"context": str, "question": ..., "history": ...}
    # 輸出 AI 的回答 (字串)
    base_llm_chain = prompt | llm | StrOutputParser()

    # 組合最終鏈 (用於 RunnableWithMessageHistory)
    # 接收 {"context": ..., "question": ..., "history": ..., "sources": ...}
    # 輸出 {"answer": str, "sources": [Docs]}
    # 使用 RunnableParallel 來同時執行 LLM (獲取答案) 並透傳 sources
    chat_output_chain = RunnableParallel(
        answer = base_llm_chain,
        sources = itemgetter("sources")
    )
    #print("chat_output_chain = " + str(chat_output_chain) + "\n")

    # (檢索) -> (格式化) -> (LLM + 透傳)
    final_rag_chain = retrieval_docs_chain | format_docs_chain | chat_output_chain

    return final_rag_chain

def main():

    # 建立 LLM
    try:
        llm = ChatOllama(model=LLM_MODEL_NAME)
        print(f"初始化成功, Model = {LLM_MODEL_NAME}")
    except Exception as e:
        print(f"錯誤：無法初始化 Ollama LLM ({LLM_MODEL_NAME})。")
        print(f"請確保 Ollama 正在運行且已拉取 (pull) '{LLM_MODEL_NAME}' 模型。")
        print(f"錯誤訊息：{e}")
        return

    retriever = get_vectorstore_retriever()

    # 建立 RAG 鏈
    rag_chain = create_rag_chain(llm, retriever)

    # 使用 RunnableWithMessageHistory 包裝 RAG 鏈
    # 這會自動處理歷史記錄的載入和儲存
    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key = "question",     # 傳遞給 rag_chain 的使用者輸入 key
        history_messages_key = "history",   # 傳遞給 rag_chain 的歷史記錄 key
        output_messages_key = "answer",     # 從 rag_chain 的輸出中，要儲存為 AI 回應的 key
    )

    print("\n --- 聊天機器人已就緒(輸入 'exit' 或 'quit 退出')")

    # 聊天循環
    session_id = "default_user_session"
    while True:
        user_input = input("you: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            continue

        start_time = time.perf_counter()
        print("機器人思考中...")
        try:
            config = {"callbacks": [langfuse_callback]}
            # 4.5. 呼叫 (invoke) 鏈
            # 需要傳遞 'question' (符合 input_messages_key)
            # 需要在 config 中傳遞 'session_id'
            config = {"configurable": {"session_id": session_id}, "callbacks": [langfuse_callback]}

            # 執行鏈
            result = chain_with_history.invoke(
                {"question": user_input},
                config = config)
            print("\nAnswer:\n" + result['answer'])

            source_docs = result['sources']
            # 顯示來源
            if source_docs:
                print("\n--- 檢索來源 ---")
                seen_sources = set()
                for doc in source_docs:
                    source_file = doc.metadata.get('source', '未知檔案')
                    page = doc.metadata.get('page', 'N/A')
                    source_key = f"{os.path.basename(source_file)} (頁碼: {page + 1})"
                    
                    if source_key not in seen_sources:
                        print(f"  - {source_key}")
                        seen_sources.add(source_key)

            print("-" * 60)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"程式執行時間: {execution_time:.4f} 秒")


        except Exception as e:
            print(f"error: {e}")
            print("-" * 60)

if __name__ == "__main__":
    main()


