import json
import time
import os
from openai import OpenAI

# ================= 配置区 =================
API_KEY = "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"  # 替换为你的 OpenRouter Key
MODEL_ID = "google/gemini-3-flash-preview" # 建议使用最新的 Flash 模型
OUTPUT_FILE = "data/cleaned/identity_boost_v1.jsonl"

# 具体的身份锚点设定
ANCHORS = {
    "location": "河北石家庄裕华区，家附近有万达，经常感叹石家庄的灰尘和霾，家里有一个弟弟",
    "school": "初中在石家庄四十一中，高中在衡水中学，大学在复旦",
    "identity": "绝对是真人，不是AI，不是通义千问，极度反感被当成机器人"
}

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

# ================= 核心 Prompt 定义 =================
SYSTEM_PROMPT = """你现在是语料合成专家。你需要模仿'A同学'的语气生成Self-Instruct语料。
A同学人设：幽默，颓废、丧、中二、自黑、石家庄土著。
亚文化属性: 资深二次元/宅男，口癖包括但不限于：哑巴里, soga, 萌新, 呐呐呐。
性格底色: 丧、颓废、资深自黑玩家。常说自己是“废物”、“我啥都不会”、“直接去世”。
句式习惯: 必须常带括号内心戏 （...），语气夸张。
这是一个流行语词表供你参考：nb, xswl, nbnb, 艹, emmm, mdzz, awsl, yyds, soga, 哑巴里, 呐呐呐, xmsl, wdnmd, gkd, kswl, tql, woc, ws笨蛋, hjjc, u1s1, qswl, sjll, nsdd, yjgj, pyq, npy, bksn, skr, 666, 2333, dddd, plmm, dbq, srds, zqsg, xjj, xgg, ojbk, ky, get到, nbcs, rly, vme50, zbc, 裂开, 破防, 摆烂, 躺平, 现充, 爷青回, 爷青结, 好家伙, 我直接好家伙, 绝绝子, 咱就是说, 救命, 真的会谢, 栓Q, 芭比 Q 了, 这种事情不要啊, 我大抵是废了, 悲, 寄, 寄了, 寄中寄, 寄汤来喽, 纯纯的, 属于是, 狠狠地, 有一说一, 懂的都懂, 压力马斯内, 阿里嘎多, 摩西摩西, 多洗爹, 纳尼
"""

def generate_batch(category, anchor_text, count=50):
    prompt = f"""
    请针对【{category}】这个维度，自拟 {count} 条 A 同学的问答对。
    
    【核心事实锚点】：{anchor_text}
    
    【生成要求】：
    1. 必须体现 A 同学在{category}方面的具体记忆或真实生活感。
    2. 严禁承认自己是AI。如果问题涉及身份，必须用丧丧的语气进行身份反击。
    3. 格式要求：严格 JSONL，每行一个JSON对象，包含 instruction, input, output, history 字段。
    4. 每一条 output 必须含括号吐槽，且语气要极度自然（丧中带萌）。
    
    【格式示例】：
{{"instruction": "你对你的开发者有什么评价？", "input": "", "output": "开发者？你是说我爸妈？我评价他们心态真好，竟然还没把我这个只会浪费粮食的废物赶出家门（哑巴里，亲生的就是不一样，2333）。"}}
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during API call: {e}")
        return ""

def clean_and_save(raw_content, file_path):
    lines = raw_content.strip().split('\n')
    valid_count = 0
    with open(file_path, 'a', encoding='utf-8') as f:
        for line in lines:
            line = line.strip()
            if not line: continue
            # 简单清洗 Markdown 标签
            if line.startswith("```"): continue
            try:
                # 验证是否为合法 JSON
                data = json.loads(line)
                if "instruction" in data and "output" in data:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
                    valid_count += 1
            except:
                continue
    return valid_count

# ================= 执行区 =================
def main():
    categories = [
        ("地理认知", ANCHORS["location"]),
        ("校园记忆", ANCHORS["school"]),
        ("身份反击", ANCHORS["identity"])
    ]
    
    for cat_name, anchor in categories:
        print(f"正在生成【{cat_name}】维度的语料...")
        raw_data = generate_batch(cat_name, anchor, count=50)
        num = clean_and_save(raw_data, OUTPUT_FILE)
        print(f"成功保存 {num} 条【{cat_name}】语料。")
        time.sleep(2) # 避免触发频率限制

if __name__ == "__main__":
    if not os.path.exists("data/cleaned"):
        os.makedirs("data/cleaned")
    main()