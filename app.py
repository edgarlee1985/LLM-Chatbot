import os
import time
from typing import Optional, Any
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

DATA_PATH = "./data"
LLM_MODEL_NAME = "qwen3"
#EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
# 使用一個支援多語言的 HuggingFace 模型，以便處理中文和英文
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

VECTOR_DB_PATH = "FAISS_DB"

def get_vectorstore_retriever() -> Optional[Any]:
    print(f"初始化 HuggingFace 嵌入模型: {EMBED_MODEL_NAME}...")
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name = EMBED_MODEL_NAME,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )

    vectorstore = []
    if os.path.exists(VECTOR_DB_PATH):
        try:
            print(f"正在從 {VECTOR_DB_PATH} 讀取向量數據庫...")
            vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization = True)
        except Exception as e:
            print(f"讀取向量數據庫時出錯: {e}")

    if not vectorstore:
        print(f"正在讀取 {DATA_PATH} 下的 **.pdf** 資料...")
        try:
            loader = DirectoryLoader(
                DATA_PATH,
                glob = "**/*.pdf",
                loader_cls = PyPDFLoader,
                show_progress = True,
                use_multithreading = True,
            )
            documents = loader.load()
        except Exception as e:
            print(f"讀取文件時出錯: {e}")
            documents = []

        if not documents:
            print("warning: 在指定路徑下未找到任何 .pdf 文件.")
            splits = []
        else:
            print(f"讀取了 {len(documents)} 頁的 pdf 檔案")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", "。", "！", "？", " ", ""],
            )
            splits = text_splitter.split_documents(documents)
            print(f"資料分割為 {len(splits)} 塊")

        if not splits:
            vectorstore = FAISS.from_texts(["初始化, 沒有資料"], embeddings)
        else:
            print("從讀取的文件建立新的 FAISS 索引...")
            vectorstore = FAISS.from_documents(splits, embeddings)
            vectorstore.save_local(VECTOR_DB_PATH)
            print(f"FAISS 索引已創建, path = {VECTOR_DB_PATH}")

    return vectorstore.as_retriever(search_kwargs = {"k": 3})

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
        docs = retriever, 
        question = RunnablePassthrough() 
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # step 1. index, 2. prompt, 3. llm, 4. output
    generation_module = (
        RunnableParallel(
            context=lambda x: format_docs(x['docs']),
            question=lambda x: x['question']
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    final_rag_chain = retrieval_module.assign(
        answer=generation_module
    ).assign(
        source_documents=lambda x: x['docs']
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
        query = input("you: ")
        if query.lower() in ["exit", "quit"]:
            break
        if not query.strip():
            continue

        start_time = time.perf_counter()
        print("機器人思考中...")
        try:
            result = rag_chain.invoke(query)
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


