# Alpaca 全量微调 7B LLaMA 模型笔记

## 数据集
- **数据源**：`alpaca_data.json`，包含 52K 条指令数据，字段包括 `instruction`、`input`、`output` 等。  
- **说明**：
  - 实际上，`instruction` 和 `input` 的目的都是让模型学会上下文 → 输出映射，因此推理时可以只给一个。
  - 训练时区分的主要原因：
    1. 让模型学会任务识别，以防模型区分不开任务意图和数据内容。
    2. 增强泛化能力，使模型能学到不同任务类型的不同处理方法。
    3. 统一模板方便批量数据处理、训练、评估。
    4. 支持复杂任务，对长文本、多段落任务有助于模型理解。
    5. 方便推理迁移。

- **发散思考**：
  - 对于长任务，如何设计 `instruction` 和 `input` 模板？（涉及具体方法后续补充）
    1. 清晰任务指令，明确具体，确保每条 `instruction` 高质量映射 `response`。
    2. 分割长文档作为 `input`。
    3. `input` 可提供上下文引用并控制信息量（如摘要、检索信息排序）。

## 数据生成
- **借用模型：** 利用当时的现进模型 `text-davinci-003`（GPT 的一个版本）生成。 
- **步骤梗概：**    
  1.构建少量示范+prompt.text给GPT，可学习prompt.txt的命令方式。few-shot意为给模型提供少量示例，充分利用GPT的模式识别能力。   
  2. 从GPT返回的文本里解析出结构化数据，拆分，过滤非法token   
  3.保存到regon.json，循环。  
- **细节研读**
   - **self-instruct原理**   
  self-instruct让指令越来越复杂，其原理是不断将模型生成的指令放入随机抽取的指令池，从而使复杂任务在池子中的比例越来越高，从而进一步生成复杂的任务，形成自举增益（Bootstrapping）。至于为什么不会滑向简单的一侧，是因为大模型本身拥有大量复杂知识，GPT3看到Instruction格式，就能够自然生成复杂任务，再加上相似度判断机制以及词数检查机制，滑向简单侧的概率非常低。
  - **encode_prompt()**   
  把若干条示例 instruction（few-shot）拼成一个带模板的 prompt，让 GPT 继续写下一条新的 instruction/input/output。相当于模式引导器   
  - **post_process_gpt3_response()**   
  把 GPT 生成的大段文本解析成结构化的 {instruction, input, output} 字典列表，同时做强力清洗和过滤。  
  问题和关键点（说白了就是GPT可能有错，人为的根据要求纠错）   
    - 分割（*由于GPT未必输出###，以这种方式分割有可能出错*）
    - 截断检测：避免最后一个不完整的输出
    - 格式检查：编号，字段，正则结构（是不是7部分组成）
    - 长度过滤：instrcution太短太长都不行
    - 语义过滤，利用黑名单，过滤掉image, graph, picture等无法完成的任务
    - 非法字符过滤：标点，非ASCⅡ开头
    - 排除重复模式：防止偏科
  - **generate_instruction_following_data()**   
  真正的生成函数（api的坑要注意）
    - 读取seed+读取历史instruction（*记得定时保存避免中途爆炸丢文件*）
    - 初始化Rouge-L来判断重复度，太高了要丢弃（*有更快的方法，数据大的时候该换直接换*）
    - 调用连接api的循环，反复生成。注意：批次检查；返回和输入形式（可以现查 知道有就行）；异常处理；

## 训练配置
- **硬件**：4 张 A100 80G  
  - 注：FSDP (Full Sharded Data Parallel)，PyTorch 的分布式训练模式，用于训练超大模型。

- **训练参数**：
  | 参数 | 设置 | 说明 |
  |------|------|------|
  | batch size | 128 | 足够大以稳定梯度，避免显存爆炸；梯度太小可能噪音大 |
  | learning rate | e-5 | 小模型可以稍大，base model 大时反而要小以稳定训练 |
  | epochs | 3-5 | 小数据集，多轮次会过拟合 |
  | max length | 512 | 保证不截断并节省显存；可对齐 batch 最长序列或按长度分组减少 padding；可用动态批处理：长序列 batch 小，短序列 batch 大 |
  | bf16 | 使用 bfloat16 半精度 | 节省显存 |
  | tf32 | 使用 TensorFloat32 | 加速矩阵运算，精度高 |
  | gradient_accum_steps | 累积步数 | 模拟大 batch size，不是每步都更新 |
  | weight decay | 0 | 权重衰减为 0，避免微调小数据集过拟合 |
  | warmup_ratio | 0.03 | 训练初期学习率逐渐升到设定值，避免梯度过大（如 10000 步训练，前 300 步线性上升，后余弦下降） |
  | scheduler | cosine | 学习率先升到最大值再慢慢减少到 0 |

## 训练函数
- **维护tokenizer**
  - 什么时候需要维护：添加新的对话角色标记；添加prompt固定格式；添加PAD, EOS, BOS token；即希望一个token表示一个概念
  - 可以根据tokenizer表来查看有无token
  - 加入特殊token可以提升微调效果，创造模型的可学习信号，告诉模型任务开始和结束的位置，给予指导从而加速模型学习
- **如何维护tokenzier**
  - 将新token加入词汇表后，用未加入前的embedding进行赋值，这样是给向量们一个合理的起点，不影响分化。