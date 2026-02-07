# 大模型量化技术概览

**高效推理与存储优化**：  
**李添翼 1217**  

---

## 目录

1. 量化背景与动机  
2. 基础概念  
3. 常用量化方法  
4. 精度评估与指标  
5. 工程思路 & 注意事项  

---

## 1. 量化背景与动机

### 1.1 背景

- 大模型参数规模大（数十亿到千亿级别），推理计算和存储压力高。
- 常见模型参数示例：

| 模型 | 架构 | 参数量 | 默认精度 |
|------|------|--------|---------|
| LLaMA 2 7B | Transformer | 7B | FP16 |
| GPT-3 | Transformer | 175B | FP16 |
| Qwen 1.5B | Transformer | 1.5B | FP16 |
| Meta LLaMA | Transformer | 7B~65B | FP16/BF16 |

- 问题：FP16 / FP32 模型推理消耗显存大，计算慢，不利于部署和加速。

### 1.2 动机

- **存储优化**：降低模型大小，例如 INT8 可节省 2 倍存储，INT4 可节省 4 倍。  
- **计算加速**：低比特整数计算能充分利用 GPU/TPU 的硬件加速能力。  
- **部署成本降低**：显存小，硬件要求低，可在更多设备上部署。  
- **激活和 KV Cache 存储压力**：在 Transformer 推理中，KV Cache 占显存大头，量化可以显著节省。  

---

## 2. 基础概念

### 2.1 量化公式

1. **量化-反量化公式（常用 RTN）**：

\[
\hat{x} = \text{Dequant}(q) = s \cdot (q - z)
\]

- \(x\)：原浮点值  
- \(q\)：量化整数值  
- \(s\)：量化步长（scale）  
- \(z\)：零点（zero-point）  
- **RTN** = Round-to-Nearest：把浮点数映射到最近整数格子  

2. **scale s 计算**：

\[
s = \frac{x_\text{max} - x_\text{min}}{q_\text{max} - q_\text{min}}
\]

- 可选按 **全张量、每列（channel-wise）、每组（group-wise）**  

3. **量化过程**：

\[
q = \text{clip}(\text{round}(x / s) + z, q_\text{min}, q_\text{max})
\]

- clip 保证整数落在允许范围内  
- FakeQuant：量化-反量化操作，前向保留浮点计算图，后向可用 STE 近似梯度  

---

### 2.2 激活、权重与 KV Cache

- **权重 W**：模型参数，如线性层和注意力矩阵  
- **激活 X**：前向计算产生的中间张量，如 linear 输入输出、FFN 中间层、softmax 前后  
- **KV Cache**：Transformer 推理时存储历史 Key / Value，用于加速多 token 生成  
  - 占显存大  
  - 不能像权重一样一次加载重复使用  

> 激活和 KV Cache 量化难度高，需要更高精度（FP16/BF16）以降低误差累积  
> 权重量化可用 INT8 / INT4，因为误差是“一次性”的  

---

### 2.3 常见精度类型

| 类型 | 描述 | 使用场景 |
|------|------|----------|
| FP32 | 单精度浮点 | 基准计算 |
| FP16 | 半精度浮点 | 推理、训练微调 |
| BF16 | 半精度浮点，指数与 FP32 相同 | 高稳定低精度训练 |
| FP8 / BF8 | 前沿训练浮点 | 超大模型训练 |
| INT8 | 8-bit 整数 | 工业标准权重量化 |
| INT4 | 4-bit 整数 | 推理低比特量化 |
| 更低比特 | 研究探索 | 权重极端压缩 |

---

## 3. 常用量化方法

### 3.1 训练后量化（PTQ）

- **核心**：不再训练权重，仅通过统计激活 / 权重分布得到 s、z  
- **方法**：
  - RTN（Round-to-Nearest）
  - GPTQ：块级量化，局部最小化 \(ΔY = XW - X \hat{W}\)
  - AWQ：激活感知权重量化，考虑输出误差加权最小化  

---

### 3.2 GPTQ

- 基于 PTQ，但允许对量化误差敏感的权重块进行 **局部微调**  
- 最小化 **输出误差 ΔY** 而非单纯权重量化误差  

---

### 3.3 AWQ

- **Activation-aware Weight Quantization**  
- 关键思想：
  1. 考虑权重在实际激活 X 下的输出贡献  
  2. 块级量化 + 按通道求最优 scale s  
  3. 对显著通道引入缩放因子 α 补偿量化误差  
- 公式：

\[
\hat{W}_j = \alpha_j \cdot \text{Quantize}(W_j / s_j) \cdot s_j
\]

- α > 1 放大量化后的显著通道，使输出接近 FP32  
- 优势：
  - 保持 INT4 / INT8 统一低比特，硬件友好  
  - 输出精度高，显著减少量化误差  

---

### 3.4 训练中量化（QAT / LSQ / PACT）

| 方法 | 核心 | 特点 |
|------|------|------|
| QAT | 在训练过程中插入 FakeQuant, 可更新权重和 scale | 精度高，可微调 s |
| LSQ | Least-Squares Quantization | 自动学习最佳 scale，使量化误差最小化 |
| PACT | Parameterized Clipping Activation | 学习激活截断值 α，减少激活量化误差 |

- FakeQuant：量化 → 反量化 → 前向计算  
- STE：反向传播时近似量化函数梯度为 1  
- 训练中量化可同时优化权重和 scale  

---

## 4. 精度评估与指标

### 4.1 任务类型指标

| 指标 | 含义 |
|------|------|
| Top-1 / Top-5 | 分类任务正确率 |
| PPL | Perplexity = exp(NLL)，衡量生成流畅度 |
| F1 | 分类 / QA 综合指标 |
| BLEU | 翻译 / 文本生成指标 |
| **MMLU** | 多任务理解能力（57 个学科选择题），对量化误差敏感 |

### 4.2 精度下降公式

\[
\Delta = \text{Metric}_{FP} - \text{Metric}_{Quant}
\]

- 若 Δ 很大 → 表示量化导致性能明显下降  
- 可通过 GPTQ / AWQ / QAT 修复  

### 4.3 层级误差指标

- **MSE**：均方误差，衡量每层输出误差  
- **Cosine Similarity**：衡量输出向量方向相似度  

---

## 5. 工程思路 & 注意事项

### 5.1 推理速度 & 显存

- **Batch 小** → latency 优先  
- **Batch 大** → throughput 优先  
- **KV Cache 占大头**，量化难度高  

### 5.2 Kernel 融合

- 将多个线性操作或激活操作合并为一个 kernel  
- 提高 memory bandwidth 利用率  
- 降低计算 overhead  

### 5.3 量化硬件加速

- INT4 / INT8 可充分利用 GPU Tensor Core / TPU INT8 单元  
- FP16 / BF16 稳定性高，误差传播可控  

---

## 6. 量化后模型实例

| 模型 | 权重量化 | 激活精度 | 模型大小 | 加速优势 |
|------|----------|----------|----------|----------|
| LLaMA 2 7B | INT4 | FP16 | 7B × 4bit ≈ 3.5B | 推理 2~4× 加速 |
| Qwen 1.5B | INT8 | FP16 | 1.5B × 8bit ≈ 1.5B | 推理 1.5~2× 加速 |
| Vicuna 13B | INT4 | FP16 | 13B × 4bit ≈ 6.5B | 推理 3~4× 加速 |
| GPT-3 175B | INT8 | FP16 | 175B × 8bit ≈ 87.5B | 推理 1.5~2× 加速 |

---

## 7. 小结与建议

1. **选择合适量化方法**：  
   - 权重：INT4 / INT8  
   - 激活 / KV Cache：FP16 / BF16  
2. **显著通道保护**：AWQ α 放大  
3. **评估指标**：MMLU 对推理精度敏感，优先关注  
4. **工程实现**：kernel 融合、batch 优化、硬件加速  
5. **策略组合**：PTQ、GPTQ、AWQ、QAT 根据精度/速度需求选择  

---

## 参考文献 / 扩展阅读

1. Dettmers et al., **GPTQ: Accurate Post-Training Quantization for Generative Pretrained Transformers**, 2023  
2. Zhao et al., **AWQ: Activation-aware Weight Quantization for Large Language Models**, 2024  
3. Jacob et al., **Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference**, 2018  
4. MMLU 数据集官网与论文：Hendrycks et al., *Measuring Massive Multitask Language Understanding*, 2020  
5. LSQ / PACT 原论文  


# INT8 / FP16 量化与 GEMM 知识总结

---

## 1. GEMM 基础

**定义**：GEMM = General Matrix Multiply（通用矩阵乘法）

数学形式：

\[
C = A \times B + C
\]

神经网络示例：

```python
y = x @ W + b  # Linear layer
```

**硬件视角**：
- GPU/CPU/TensorCore/NNPU 通常把 Linear / Attention 等操作最终转成 GEMM

**类型对比**：

| 类型 | 输入 dtype | 乘法单元 | 累加 | 精度 | 特点 |
|------|------------|-----------|------|------|------|
| FP16 GEMM | FP16 | FP16 Tensor Core | FP16/FP32 | 高 | 稳定、吞吐高 |
| INT8 GEMM | INT8 | INT8 Tensor Core / DP4A | INT32 | 较低 | 吞吐更高、对 scale 敏感 |

**注意**：FP16 / INT8 只是 GEMM 的数值表示和硬件实现不同，本质是同一个 GEMM。

---

## 2. 权重量化与激活量化

### 2.1 权重量化

- 权重 W 被量化成 INT8 存储
- 在推理时 **先 cast 成 FP16 再计算**
- 数学形式：

\[
W_{fp16} = W_{int8} \times scale
\]

- 特点：
  - 误差固定，可控
  - 对任意输入 x，误差线性依赖 x
  - 可以离线统计和优化

### 2.2 激活量化

- 输入激活 x 如果被量化，误差随输入而变化
- 数学形式：

\[
x_{fp16} = x_{int8} \times scale
\]

- 特点：
  - 输入依赖，分布动态变化
  - 误差不可控，容易在多层累积被放大
  - 高风险，特别是 KV cache / Transformer Attention

### 2.3 权重 vs 激活误差对比

| 类型 | 误差性质 | 可控性 |
|------|------------|--------|
| 权重量化 | 静态，固定矩阵 | 高，可校准 |
| 激活量化 | 动态，依赖输入 | 低，难以校准 |

---

## 3. INT8 → FP16 转换（cast）

### 3.1 核心概念

- **不是补 0 / 扩展位宽**
- **是数值重解释**：整数值先被理解，再重新编码成浮点
- 在硬件上由组合逻辑完成，几乎是零开销

### 3.2 具体步骤

```text
int8 value (bits) --> 解码为整数 N --> 用 FP16 规则重新编码 N --> 得到 FP16 数值
```

- 例：
  - int8 = 38 → FP16 = 38.0
  - int8 = -26 → FP16 = -26.0

- 对 int8 范围 [-128,127]，FP16 能精确表示，不会丢信息

### 3.3 与 scale 的关系

- 实际量化计算：

```text
FP16_weight = FP16(int8_value) * scale
```

- cast 与乘 scale 是两步，但在硬件流水线上可以融合

### 3.4 工程心智模型

> int → float 是“改变解释规则 + 硬件电路直出结果”，不是存内存再算，也不改 bits

---

## 4. 数值误差展开（数学直觉）

一层 Linear：

\[
y = (W + \Delta W)(x + \Delta x) = Wx + W\Delta x + \Delta W x + \Delta W \Delta x\]

- **ΔW**：权重量化误差，固定，可控
- **Δx**：激活量化误差，动态，风险高
- **ΔW·Δx**：权重 + 激活同时量化时的高阶误差，可能被层叠放大

---

## 5. 总结

1. **GEMM**：通用矩阵乘法，FP16 / INT8 是不同数据类型和硬件实现路径
2. **权重量化可控**：固定误差，线性依赖输入
3. **激活量化风险高**：输入依赖、动态分布、不易校准
4. **INT8 → FP16 cast**：硬件组合逻辑完成数值重解释 + scale，几乎零开销
5. **误差展开**：线性可控 vs 高阶累积风险

> 一句话总结：
> 权重量化静态可控，激活量化动态不可控；INT8 → FP16 是硬件内建指令，完成数值重解释。

---