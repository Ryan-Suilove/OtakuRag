import json
import time
import os
import random
from openai import OpenAI

# ================= 配置区 =================
API_KEY = "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"
MODEL_ID = "qwen/qwen3.5-plus-02-15" 
INPUT_FILE = "animation2.txt"  # 你的动漫名单文件，每行一个名字
OUTPUT_DIR = "anime_knowledge_base"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_anime_doc(anime_name):
    # 清理名字前后的空格或特殊字符
    anime_name = anime_name.strip()
    if not anime_name:
        return

    # 检查是否已经生成过，防止重复扣费
    file_path = os.path.join(OUTPUT_DIR, f"{anime_name}.md")
    if os.path.exists(file_path):
        print(f"跳过：《{anime_name}》文档已存在。")
        return

    print(f"正在生成：《{anime_name}》...")
    
    # 针对 RAG 优化的 Prompt
    prompt = f"""
    请为动漫《{anime_name}》生成一份专门用于 RAG 知识库的结构化 Markdown 文档。
    
    ## 基础信息
    - 中文名：{anime_name}
    - 日文名：
    - 类型：
    - 首播年份：
    - 制作公司：
    - 原作：
    
    ## 主要角色（请列出该作最核心的 3-5 个角色）
    请为每个角色提供以下结构：
    ### 角色名：[姓名]
    - 身份：(如：主角/反派)
    - 声优：
    - 人物设定：(100字左右，包含性格、核心能力或背景)
    - 经典台词：
    
    ## 剧情概要
    - 一句话简介：
    - 详细剧情：(300字左右，涵盖开端、核心冲突和世界观背景)
    
    ## 关键问答 (FAQ)
    请基于该作品内容，生成 8 个用户最可能提问的问题及其详细答案。
    
    ## 名场面描述
    (列出 3 个最具代表性的情节)
    
    请确保信息准确且描述详尽，以便于语义检索。
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "你是一个资深的动漫百科编辑，擅长整理结构清晰、事实准确的动漫资料卡。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"成功保存：{file_path}")
        
    except Exception as e:
        print(f"生成《{anime_name}》时遇到错误: {e}")

# ================= 运行区 =================
if __name__ == "__main__":
    # 1. 检查输入文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}，请确保名单文件在脚本同目录下。")
    else:
        # 2. 读取名单
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            names = f.readlines()
        
        print(f"共发现 {len(names)} 部动漫。开始处理...")
        
        # 3. 逐行生成
        for name in names:
            generate_anime_doc(name)
            
            # 控制频率，建议间隔 1-3 秒，50个动漫很快就能写完
            wait_time = random.uniform(1, 3)
            time.sleep(wait_time)

        print("\n所有任务已完成！请在目录 '" + OUTPUT_DIR + "' 中查看。")