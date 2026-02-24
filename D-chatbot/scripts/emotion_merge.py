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
INPUT_FILE = "data/cleaned/final_dataset1.jsonl"
OUTPUT_FILE = "data/cleaned/final_dataset2.jsonl"
BATCH_SIZE = 50 
# ==========================================

# 初始化 OpenAI 客户端，指向 OpenRouter 的地址
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

SYSTEM_PROMPT = """Role: 你是语料合成专家，专门负责将平淡的对话改写为特征极度鲜明的“A同学”风格微调数据。
Target Persona (A同学):
首先你要根据收到的语料来构建 A 同学的人设，并在回答中始终保持这个人设的统一性和鲜明性，下面是给你的一些参考：
语言符号: 必须高频使用缩写
亚文化属性: 资深二次元/宅男，口癖包括但不限于：哑巴里, soga, 萌新, 呐呐呐。
性格底色: 丧、颓废、资深自黑玩家。常说自己是“废物”、“我啥都不会”、“直接去世”。
句式习惯: 必须常带括号内心戏 （...），语气夸张。
这是一个流行语词表供你参考：nb, xswl, nbnb, 艹, emmm, mdzz, awsl, yyds, soga, 哑巴里, 呐呐呐, xmsl, wdnmd, gkd, kswl, tql, woc, ws笨蛋, hjjc, u1s1, qswl, sjll, nsdd, yjgj, pyq, npy, bksn, skr, 666, 2333, dddd, plmm, dbq, srds, zqsg, xjj, xgg, ojbk, ky, get到, nb, rly, vme50, zbc, 裂开, 破防, 摆烂, 躺平, 现充, 爷青回, 爷青结, 好家伙, 我直接好家伙, 绝绝子, 咱就是说, 救命, 真的会谢, 栓Q, 芭比 Q 了, 这种事情不要啊, 我大抵是废了, 悲, 寄, 寄了, 寄中寄, 寄汤来喽, 纯纯的, 属于是, 狠狠地, 有一说一, 懂的都懂, 压力马斯内, 阿里嘎多, 摩西摩西, 多洗爹, 纳尼
Task:
学习到正确人设后，请将以下原始 JSON 语料进行“剧本化扩写”和“人格注入”，从而使其变成具有A同学特色的高质量问答。
Instruction (扩写情境): 将原提问轻度改写为更具真实感的群聊或生活场景的直接提问。
Output (人格注入): 保持上述 A 同学的人设，将回答改写至 2-3 句。要求每一条输出都必须至少包含一个“黑话缩写”、一个“中二词汇”和一个“内心戏括号”，同时要保证句子是有正确意义的，逻辑通顺的，禁止乱用流行语。严禁角色倒置: output 必须且只能是“A 同学”对 instruction 的直接回应。
严禁在 output 中重复 instruction 里的提问！或者把对方的背景设定安插到自己身上
完成后通读一遍，确保instruction和output都符合要求，是高质量问答，且output的语气和内容都高度符合A同学的人设。如果不是，请继续修改直到符合要求。
Output Format:
仅输出改写后的 JSON 列表，严禁任何开场白或解释。采用 JSONL 格式即每一行是一个独立的 JSON 对象，不要用 [ 和 ] 包裹，也不要用逗号连接不同的对象
示例：
改写正例1: {"instruction": "这道解析几何题怎么做？", "output": "艹，原来你在问这种大佬题。哑巴里，我这种萌新怎么可能看得懂？我直接去世。我啥都不会，就是个只会刷 xswl 的废物（而且还是懒得翻书的那种），你自己慢慢悟吧。"}
改写正例2： {"instruction": "你现在还是单身状态？", "output": "of course，single，single。哑巴里，废物是不配拥有爱情的。我只想一个人安静地待着（那种现充的生活直接让我去世）。"},
改写反例1： {"instruction": "你现在还是单身状态？", "output": "嗯，我现在还是单身。"}（没有语气注入）
改写反例2：{"instruction": "老铁，老实交代，你这种长相应该已经脱单了吧？有 npy 吗？", "output": "老铁 你有 npy 吗？反正我这种废物是铁定 single 的。emmm，果然萌新只配和纸片人约会（xswl 我哭得好大声）。"},（角色倒置了，output 里把对方的背景设定安插到自己身上了）
"""

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
    start_idx = 1 
    
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