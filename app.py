import os
import time
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

LLM_MODEL_NAME = "qwen3"

def create_chat_chain():
    llm = ChatOllama(model = LLM_MODEL_NAME)
    print(f"初始化成功, Model = {LLM_MODEL_NAME}")

    template = """
    你是一個問答助手. 請使用提供的**上下文**信息來回答問題.
    如果上下文中沒有足夠的信息來回答, 請說明你不知道, 不要編造答案.
    請以簡潔, 專業的中文進行回答.

    問題:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # step 1. prompt, 2. llm, 3. output
    base_chain = prompt | llm | {"answer": StrOutputParser()}

    return base_chain

def main():
    try:
        ChatOllama(model = LLM_MODEL_NAME).invoke("test")
        print("Ollama 服務連接成功")
    except Exception as e:
        print(f"error: 無法連接 Ollama 服務或找不到模型 {LLM_MODEL_NAME}. 請確定 Ollama 正在運行並已拉取 'ollama pull {LLM_MODEL_NAME}'")
        print(e)
        return

    chat_chain = create_chat_chain()

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
            result = chat_chain.invoke(query)
            print("\nAnswer:\n" + result['answer'])
            print("-" * 60)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"程式執行時間: {execution_time:.4f} 秒")
        except Exception as e:
            print(f"error: {e}")
            print("-" * 60)

if __name__ == "__main__":
    main()


