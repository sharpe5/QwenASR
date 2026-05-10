# QwenASR 性能优化清单

本文档汇总了 QwenASR 纯 Rust CPU 推理引擎（在 Apple M5 上达 41+ RTF）中使用的核心优化技术。

## 1. 内存带宽与分配优化

- **Decoder INT8 量化**: Decoder 权重（`QKV`、`O` 投影、`FFN` gate/up/down、`lm_head`）在加载时量化为带逐行缩放因子的 INT8。相比 FP32，内存读写减少约 4 倍，并去除了 BF16 到 F32 的转换开销。由专用 NEON INT8 算子执行。
- **可复用工作区**: 消除了热点路径的堆内存分配。
  - **Encoder**: `EncoderBuffers` 持久化 `chunk_mel`、卷积变量和 `im2col` 矩阵。主激活张量 (`x`) 和 `window_starts` 在多次调用间复用。
  - **Decoder**: `DecoderBuffers` 预分配 BF16 到 F32 的转换空间，每次 Prefill 消除约 140 次分配。
  - **Transcription**: 流式模式直接复用 Embedding 组装缓冲区。
- **静态权重预转换**: 多 Token Decoder Prefill 权重在加载时从 BF16 转换为 F32 并缓存，避免流式或分段推理时重复转换。

## 2. 算子融合与缓存局部性

- **融合残差累加**: 移除独立的 `y = y + x` 循环。通过 `linear_accumulate` 和 `linear_nobias_bf16_addto` 算子，矩阵乘结果直接累加到残差目标缓冲区，每层减少一次读写。
- **融合 Matvec + SwiGLU**: 使用融合算子在计算 `gate_up` 投影的同时立即执行 `SwiGLU` 激活，使中间结果保持在 L1 缓存中。
- **Head 连续的 KV Cache**: KV Cache 内存布局改为 `[layer][head][pos][head_dim]`。在 Causal Attention 按 Head 遍历历史位置时，连续内存大幅降低 Cache Miss。

## 3. SIMD 与平台加速

- **显式 SIMD 内联汇编**:
  - `rms_norm`、`gelu` 和 `swiglu` 激活函数向量化，并使用多项式逼近指数运算。
  - RoPE 旋转位置编码使用 NEON 向量指令成对计算。
  - 使用 `vshll_n_u16` (NEON) 和 `_mm256_cvtepu16_epi32` (AVX2) 批量转换 BF16。
- **Apple Accelerate & vDSP**: 密集矩阵运算（注意力分数、Mel 频谱生成）卸载至 Accelerate (BLAS)。使用 `vvexpf` 批量计算 Softmax 指数，`vDSP_dotpr` 触发 AMX 协处理器。

## 4. 线程与并发

- **无锁线程池快速路径**: 任务调度优先使用原子操作和自旋等待，失败后再休眠，降低细粒度并行任务（如多头注意力）的系统调度延迟。
- **非矩阵运算并行化**:
  - Encoder 卷积的 `im2col` 数据打包。
  - 大型 FFN 缓冲区上的 `gelu` 和 `swiglu` 计算。
  - 跨 Head 的双向注意力计算。

## 5. 算法级优化

- **静音压缩**: 基于能量的 VAD 预处理剥离非语音片段。边缘填充降至 2 个窗口，并消除非语音拖尾，直接减少 Encoder 和 Decoder 输入。
- **流式惰性编码**: 流式模式下，局部的 Encoder 尾部仅每隔一个 Chunk 重新编码。提供最长公共前缀 (LCP) 复用，跳过重新编码的 Chunk 的 Decoder Prefill 开销降低约 50%。
- **在线 Softmax**: 单 Token Causal Attention 使用在线 Softmax 扫描，在单次循环内完成分数追踪、归一化和值累加，避免为 `seq_len = 1` 场景分配临时数组和二次遍历。
