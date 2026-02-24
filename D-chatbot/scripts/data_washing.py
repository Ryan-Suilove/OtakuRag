import json
import re

# ================= 配置区 =================
INPUT_FILE = "data/flitered/version1.json"      # 原始数据
OUTPUT_FILE = "data/cleaned/outputs_clean.jsonl"  # 清洗后的输出
# ==========================================

# 正则规则
LINK_PATTERN = re.compile(r"https?://\S+")
ONLY_NUM_SYMBOL_PATTERN = re.compile(r"^[\d\W_]+$")
EMOJI_PROMPT_PATTERN = re.compile(r"^\[.*?\]请使用最新版手机QQ体验新功能$")
RED_PACKET_PATTERN = re.compile(r"^\[QQ红包\]请使用新版手机QQ查收红包$")

# 去掉 @username
def remove_at_username(text):
    return re.sub(r"@\S+", "", text).strip()

# 去掉句首句尾双引号
def remove_edge_quotes(text):
    return text.strip('"').strip()

def is_valid_line(output):
    """判断一条 output 是否有效"""
    output = output.strip()
    if not output:
        return False
    if LINK_PATTERN.search(output):
        return False
    if ONLY_NUM_SYMBOL_PATTERN.match(output):
        return False
    if EMOJI_PROMPT_PATTERN.match(output):
        return False
    if RED_PACKET_PATTERN.match(output):
        return False
    return True

def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line)
                output = data.get('output', '')
                output = remove_at_username(output)
                output = remove_edge_quotes(output)
                if is_valid_line(output):
                    data['output'] = output
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                continue

    print(f"清洗完成，结果已保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
