from rag_enginev2 import RAGEngineV2

def start_interactive_chat():
    try:
        # 初始化双路检索引擎
        engine = RAGEngineV2()
        print("\n--- A同学已上线！(双路检索模式：FAISS + Neo4j) ---")
        print("--- 输入 'exit' 或 'quit' 退出 ---\n")

        while True:
            user_input = input("你: ")
            if user_input.lower() in ['exit', 'quit']:
                break

            response = engine.invoke(user_input)
            print(f"\n助手: {response}\n")

    except Exception as e:
        print(f"启动失败: {e}")
    finally:
        # 确保关闭连接
        if 'engine' in locals():
            engine.close()

if __name__ == "__main__":
    start_interactive_chat()