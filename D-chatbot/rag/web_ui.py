"""
Gradio Web UI - A同学 AI 助手
"""

import gradio as gr
import sys
import os

# 添加 rag 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_enginev2 import RAGEngineV2

# 全局引擎实例（避免每次对话都重新加载）
engine = None


def get_engine():
    """获取或初始化引擎"""
    global engine
    if engine is None:
        print("[INFO] 正在初始化 RAG 引擎...")
        engine = RAGEngineV2()
        print("[INFO] 引擎初始化完成！")
    return engine


def chat(message, history):
    """
    聊天函数
    message: 用户当前输入
    history: 历史对话记录 [[用户消息, 机器人回复], ...]
    """
    if not message.strip():
        return ""

    try:
        rag_engine = get_engine()
        response = rag_engine.invoke(message)
        return response
    except Exception as e:
        return f"抱歉，出错了：{str(e)}"


# 创建界面
def create_interface():
    with gr.Blocks(
        title="A同学 AI 助手"
    ) as demo:
        gr.Markdown("""
        # A同学 AI 助手
        精通二次元知识的废宅风AI助手，随便聊聊吧~
        """)

        chatbot = gr.ChatInterface(
            fn=chat,
            chatbot=gr.Chatbot(
                height=500,
                show_label=False,
                avatar_images=(None, "A")
            ),
            textbox=gr.Textbox(
                placeholder="输入你的问题...",
                show_label=False,
                container=False,
            ),
        )

        gr.Markdown("""
        ---
        *提示：本助手基于 FAISS 向量库 + Neo4j 知识图谱，回答可能需要几秒钟加载*
        """)

    return demo


if __name__ == "__main__":
    print("=" * 50)
    print("  A同学 AI 助手 - Web 界面启动中...")
    print("=" * 50)

    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,  # 设为 True 可生成公网链接
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
        )
    )