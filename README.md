# A同学 AI 助手

基于 Qwen2.5 微调的二次元风格 AI 聊天助手，结合 FAISS 向量检索与 Neo4j 知识图谱实现 RAG 增强问答。

## 项目简介

A同学是一个具有鲜明人格设定的 AI 助手：
- **人设定位**：精通二次元知识的废宅风 AI，语气幽默、颓废、中二
- **核心能力**：动漫知识问答、角色剧情解析、作品推荐
- **技术特色**：双路并行检索策略，结合向量检索与知识图谱

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 基座模型 | Qwen2.5 |
| 微调框架 | LlamaFactory (LoRA) |
| 模型量化 | Llama GGUF |
| 向量数据库 | FAISS |
| 知识图谱 | Neo4j |
| Embedding | shibing624/text2vec-base-chinese |
| 推理服务 | LM Studio |
| Web 界面 | Gradio |

## 项目结构

```
RyanProjects/
├── D-chatbot/                    # 主项目目录
│   ├── data/                     # 数据目录
│   │   ├── raw/                  # 原始聊天记录
│   │   ├── cleaned/              # 清洗后数据
│   │   ├── flitered/             # 过滤后数据
│   │   └── train/                # 训练数据集 (JSON)
│   ├── scripts/                  # 数据处理脚本
│   │   ├── data_construct.py     # 训练数据构造
│   │   ├── data_washing.py       # 数据清洗
│   │   └── emotion_merge.py      # 情感数据合并
│   ├── rag/                      # RAG 检索服务
│   │   ├── rag_enginev2.py       # 双路检索引擎
│   │   ├── build_index_v2.py     # FAISS 索引构建
│   │   ├── auto_graph_builder.py # Neo4j 知识图谱构建
│   │   ├── web_ui.py             # Gradio Web 界面
│   │   └── faiss_index_v2/       # FAISS 索引文件
│   └── wiki/                     # 动漫知识库
│       ├── wiki_process.py       # 知识库生成脚本
│       └── anime_knowledge_base/ # 50+ 动漫 Markdown 文档
├── Notes/                        # 学习笔记
└── Alpaca/                       # 微调学习记录
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/Ryan-Suilove/RyanProjects.git
cd RyanProjects/D-chatbot

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动推理服务

使用 LM Studio 加载量化后的 GGUF 模型，启动本地 API 服务（默认端口 1234）。

### 3. 启动 Neo4j（可选）

```bash
# Docker 启动 Neo4j
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/12345678 \
  neo4j:latest
```

### 4. 构建 RAG 索引

```bash
cd D-chatbot/rag

# 构建 FAISS 向量索引
python build_index_v2.py

# 构建 Neo4j 知识图谱（需先启动 Neo4j）
python auto_graph_builder.py
```

### 5. 启动 Web 服务

```bash
python web_ui.py
```

访问 `http://127.0.0.1:7860` 即可与 A同学 对话。

## 训练数据构造

### 数据格式

采用 Alpaca 格式的 instruction/input/output 结构：

```json
{
  "instruction": "你是谁？",
  "input": "",
  "output": "我就是那个在石家庄摆烂的 A 同学啊...",
  "system": "你是一个叫A同学的，精通二次元知识的，自嘲废宅风的AI助手"
}
```

### 数据处理流程

```
原始聊天记录 → 数据清洗 → 上下文筛选 → 身份锚点注入 → 训练数据集
```

核心脚本：
- `scripts/data_washing.py` - 清洗无意义内容
- `scripts/data_construct.py` - 构造 Self-Instruct 语料

## RAG 检索架构

### 双路并行检索策略

```
用户问题
    │
    ▼
┌─────────────┐
│  jieba 分词  │
└─────────────┘
    │
    ├──────────────────┐
    ▼                  ▼
┌─────────┐      ┌──────────┐
│  FAISS  │      │  Neo4j   │
│ 向量检索 │      │ 图谱检索  │
└─────────┘      └──────────┘
    │                  │
    └──────────┬───────┘
               ▼
         上下文融合
               │
               ▼
         LLM 生成回答
```

### 知识库规模

- **动漫数量**：50+ 部热门作品
- **文档格式**：结构化 Markdown（角色信息、剧情概要、FAQ、名场面）
- **向量索引**：基于 text2vec-base-chinese 语义嵌入

## 微调配置

使用 LlamaFactory 进行 LoRA 微调：

```yaml
# 示例配置
model_name: Qwen/Qwen2.5-7B-Instruct
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
learning_rate: 5e-5
num_train_epochs: 3
batch_size: 4
```

## 模型量化

使用 llama.cpp 将微调后的模型转换为 GGUF 格式：

```bash
# 转换为 GGUF
python convert-hf-to-gguf.py ./output_model --outfile a_tongxue.gguf

# 量化 (可选 Q4_K_M, Q5_K_M, Q8_0 等)
./llama-quantize a_tongxue.gguf a_tongxue-q4_k_m.gguf Q4_K_M
```

## 效果展示

```
用户：鲁迪乌斯是谁？

A同学：纳尼？你竟然不知道鲁迪乌斯？这可是《无职转生》的男主啊（虽然是个废柴转生异世界的故事，但是意外地带感）。
全名鲁迪乌斯·格雷拉特，前世是个34岁的无职宅男，转生后保留了前世记忆，从婴儿开始重新做人。
soga，虽然设定听起来有点怪，但这部番绝对是异世界天花板级别的存在，懂的都懂。
```

## 致谢

- [Qwen](https://github.com/QwenLM/Qwen2.5) - 阿里通义千问基座模型
- [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) - 高效微调框架
- [LangChain](https://github.com/langchain-ai/langchain) - RAG 开发框架
- [FAISS](https://github.com/facebookresearch/faiss) - 向量检索库

## License

MIT License