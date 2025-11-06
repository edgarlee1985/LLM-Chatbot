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

from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

DATA_PATH = "./data"
LLM_MODEL_NAME = "qwen3"
#EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
# 使用一個支援多語言的 HuggingFace 模型，以便處理中文和英文
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

VECTOR_DB_PATH = "FAISS_DB"


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

def create_rag_chain(retriever):
    llm = ChatOllama(model = LLM_MODEL_NAME)
    print(f"初始化成功, Model = {LLM_MODEL_NAME}")

    template = """
    你是一個問答助手. 請使用提供的**上下文**信息來回答問題.
    如果上下文中沒有足夠的信息來回答, 請說明你不知道, 不要編造答案.
    請以簡潔, 專業的中文進行回答.
    上下文:
    {context}

    問題:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_module = RunnableParallel(
        docs = RunnableLambda(lambda x: x['question']) | retriever, 
        question = lambda x: x['question']
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # step 1. index, 2. prompt, 3. llm, 4. output
    generation_module = (
        RunnableParallel(
            context = lambda x: format_docs(x['docs']),
            question = lambda x: x['question']
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    final_rag_chain = retrieval_module.assign(
        answer = generation_module
    ).assign(
        source_documents = lambda x: x['docs']
    ).pick(["answer", "source_documents"])

    return final_rag_chain

def main():
    try:
        ChatOllama(model = LLM_MODEL_NAME).invoke("test")
        print("Ollama 服務連接成功")
    except Exception as e:
        print(f"error: 無法連接 Ollama 服務或找不到模型 {LLM_MODEL_NAME}. 請確定 Ollama 正在運行並已拉取 'ollama pull {LLM_MODEL_NAME}'")
        print(e)
        return

    retriever = get_vectorstore_retriever()

    rag_chain = create_rag_chain(retriever)

    print("\n --- 聊天機器人已就緒(輸入 'exit' 或 'quit 退出')")

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
            result = rag_chain.invoke(
                {"question": user_input},
                config = config)
            print("\nAnswer:\n" + result['answer'])

            source_docs = result['source_documents']
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


