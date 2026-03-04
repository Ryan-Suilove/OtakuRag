from rag_engine import RAGEngine

def start_interactive_chat():
    try:
        # 初始化引擎
        engine = RAGEngine()
        print("\n--- A同学已上线！(输入 'exit' 退出) ---")
        
        while True:
            user_input = input("你: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            response = engine.invoke(user_input)
            print(f"\n助手: {response}\n")
            
    except Exception as e:
        print(f"启动失败: {e}")

if __name__ == "__main__":
    start_interactive_chat()