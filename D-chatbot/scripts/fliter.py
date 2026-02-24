import re
#本脚本用于从原始聊天记录中提取指定QQ号的消息，并过滤掉图片和表情等非文本内容。最终结果将保存到新的文本文件中，便于后续分析和处理。
INPUT_FILE = "data/raw/xrjb.txt"
OUTPUT_FILE = "data/cleaned/filtered_chat.txt"
TARGET_QQ = "480667648"

# 匹配：时间 昵称(QQ号)
header_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) .*\((\d+)\)$")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

result = []

i = 0
while i < len(lines) - 1:
    header = lines[i].strip()
    content = lines[i + 1].strip()

    match = header_pattern.match(header)
    if match:
        qq = match.group(2)

        # 1️⃣ QQ号必须匹配
        if qq == TARGET_QQ:
            # 2️⃣ 内容过滤
            if content and "[图片]" not in content and "[表情]" not in content:
                result.append(header + "\n")
                result.append(content + "\n\n")

        i += 2
    else:
        i += 1  # 防御性跳过异常行

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(result)

print("提取完成，共保存 {} 条消息".format(len(result) // 2))
