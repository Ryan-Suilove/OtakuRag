import json

input_file = "data/cleaned/filtered_chat.txt"
output_file = "data/cleaned/clean.json"

results = []

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    
    # 判断是否是时间行（简单判断：以年份开头）
    if line.startswith("20"):  
        # 下一行是内容
        if i + 1 < len(lines):
            content = lines[i + 1].strip()
            if content:  # 防止空内容
                results.append({"output": content})
        i += 3  # 跳过 时间行 + 内容行 + 空行
    else:
        i += 1

# 写入 jsonl 文件
with open(output_file, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"处理完成，共保留 {len(results)} 条语料")
