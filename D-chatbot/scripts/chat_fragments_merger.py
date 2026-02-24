import json
import time
import os
import random
from openai import OpenAI  # 必须使用 OpenAI 库来对接 OpenRouter
#这个脚本的目的是将 A 同学的聊天碎片进行语义合并和清理，生成更连贯的对话数据集。它使用 OpenRouter 上的 Gemini 3 Flash 模型来处理批量的聊天碎片，并且包含了自动断点续传和错误处理机制，以确保在遇到服务器问题或账户问题时能够安全地停止或继续运行。
# ================= 配置区 =================
# OpenRouter API Key
API_KEY = "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"
# OpenRouter 上的 Gemini 3 Flash 模型 ID
MODEL_ID = "google/gemini-3-flash-preview" 
INPUT_FILE = "data/cleaned/fixup.jsonl"
OUTPUT_FILE = "data/cleaned/final_dataset1.jsonl"
BATCH_SIZE = 50 
# ==========================================

# 初始化 OpenAI 客户端，指向 OpenRouter 的地址
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

SYSTEM_PROMPT = """你是一个对话语料重构专家。我会提供 A 同学的聊天碎片。请执行以下逻辑：
1. **语义合并**：观察连续的 Output，如果它们在话题上是延续的（如：指代关系“他/它”、动作描述“笑笑说”、明显的在讲一件事），请将它们合并，允许且只允许添加必要的标点符号，严格禁止修改破坏原有语料内容。
2. **清理废话**：删除无实际意义的占位符（如“说”、“他说”）。
3. **反推 Input**：为合并后的 Output 遵循以下规则编写一个最合理的对话 Input。
Strategies (必须严格遵守):
直接对话型：如果 Output 语义明确，直接生成对应的询问或话题发起。
轻度重构型：如果 Output 只有“无聊”、“好吧”等较难发起对话的内容，允许对 Input 进行轻度重构（例如将“无聊”反推为“你在干嘛”或“我好累”），使其成为一个有意义的对话开端。
极短接话型：如果 Output 是纯情绪化或无意义符号（如“？”、“...”），保持 Input 同样为极短的互动，保留生活化质感。
4. **格式要求**：严格输出 JSONL 格式，每行包含 {"input": "...", "output": "..."}。不要输出任何解释说明。"""

def process_batch(lines, batch_info):
    content = "\n".join([json.loads(l)['output'] for l in lines])
    
    retry_count = 0
    while True:
        try:
            # 使用 OpenAI 兼容模式调用
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"请处理以下语料块：\n{content}"}
                ],
                # 针对 OpenRouter 的额外配置（可选）
                extra_headers={
                    "HTTP-Referer": "http://localhost:3000", 
                    "X-Title": "Chat Fragment Merger",
                }
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            err_str = str(e).upper()
            
            # 1. 处理服务器繁忙或超时
            if "503" in err_str or "OVERLOADED" in err_str or "TIMEOUT" in err_str:
                wait_time = min(2 ** retry_count + random.random(), 60)
                print(f"服务器忙/超时，等待 {wait_time:.1f}s 后重试...")
                time.sleep(wait_time)
                retry_count += 1
                continue
                
            # 2. 处理欠费、配额或 API Key 问题
            elif "429" in err_str or "BALANCE" in err_str or "INSUFFICIENT" in err_str or "401" in err_str:
                print(f"\n[停止运行] 账户问题（余额不足/Key无效）: {e}")
                print(f"当前中断位置：第 {batch_info} 条数据附近。")
                return "STOP_SIGNAL"
                
            # 3. 其他致命错误
            else:
                print(f"遇到无法自动修复的错误: {e}")
                return None

def main():
    if not os.path.exists(INPUT_FILE):
        print("错误：找不到输入文件")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    # --- 自动断点检测逻辑 ---
    # 如果要从 1050 条开始，请手动修改 start_idx = 1050
    start_idx = 751 
    
    print(f"OpenRouter 链路就绪。开始处理，总计 {len(all_lines)} 条数据...")

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        for i in range(start_idx, len(all_lines), BATCH_SIZE):
            batch = all_lines[i:i+BATCH_SIZE]
            print(f"--- 正在处理第 {i} 到 {min(i+BATCH_SIZE, len(all_lines))} 条 ---")
            
            result = process_batch(batch, i)
            
            if result == "STOP_SIGNAL":
                break
                
            if result:
                # 过滤掉模型可能输出的 Markdown 代码块标识
                clean_result = result.replace("```json", "").replace("```jsonl", "").replace("```", "").strip()
                f_out.write(clean_result + "\n")
                f_out.flush()
            
            # 付费后延迟可以降低，1-2秒即可
            time.sleep(1.5)

if __name__ == "__main__":
    main()