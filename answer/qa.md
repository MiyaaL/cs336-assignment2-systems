# CS336 Spring 2025 Assignment 2 Systems - QA（中文）

> 说明：本文件按 handout 目录组织，覆盖 PDF 中全部 QA 问题类型（profiling/benchmark/mixed precision/attention/compile/DDP/sharding）。

---

## 1.1 Profiling and Benchmarking

### 1.1.1 Setup - Importing your Basics Transformer Model

**Q：如何验证 assignment1 的模型可被 systems 工程复用？**

A：
- 保证顶层 `pyproject.toml` 正确引用 `cs336-basics`。
- 在环境内能成功 `import cs336_basics` 即可。
- 本仓库提供 `benchmark.py` 直接实例化 `BasicsTransformerLM`，说明接线已完成。

### 1.1.2 Model Sizing

**Q：small/medium/large/xl/2.7b 配置如何映射？**

A：
`cs336_systems/benchmark.py` 的 `MODEL_SIZES` 已给出：
- small: `d_model=768, d_ff=3072, num_layers=12, num_heads=12`
- medium: `1024, 4096, 24, 16`
- large: `1280, 5120, 36, 20`
- xl: `1600, 6400, 48, 25`
- 2.7b: `2560, 10240, 32, 32`

### 1.1.3 End-to-End Benchmarking

**Q(a)：benchmark 脚本实现在哪里？**

A：`cs336_systems/benchmark.py`。

**Q(b)：forward/backward 时间关系？**

A：通常 backward 显著慢于 forward（常见约 1.5x~2.5x），因为需要额外反向图计算与梯度写回。

**Q(c)：warm-up 的作用？**

A：有 warm-up 时，首次 kernel 编译/缓存建立/内存页映射成本不会污染计时，结果更稳定。

### 1.1.4 Nsight Systems Profiler

**Q(a)：`nvtx.range` 是否有帮助？**

A：有。可以明确标注区间边界，便于区分 forward/backward/optimizer.step 的 kernel 归属。

**Q(b)：热点算子是什么？**

A：典型热点是 GEMM（QK^T、PV、MLP 线性层），其次是 softmax 与若干 elementwise kernel。

**Q(c)：为什么调用次数不一定等于迭代次数？**

A：一次迭代会拆分成多 kernel（含不同 shape、不同算子阶段），且框架会插入额外内核（sync/copy/reduction）。

**Q(d)：forward-only 与 full-train 的 profile 差异？**

A：forward-only 里 GEMM 占比更集中；full-train 会增加大量 backward kernel 与 optimizer 相关开销。

**Q(e)：softmax FLOPs 占比低但耗时不一定低，为什么？**

A：softmax 常受内存带宽与访存模式限制，算强度低，容易变成 memory-bound。

### 1.1.5 Mixed Precision

**Q(a)：autocast 下哪些算子会保留 fp32？**

A：LayerNorm/RMSNorm 一类归一化通常保持 fp32 累加以稳住数值；线性层常走 fp16/bf16 tensor core。

**Q(b)：为何归一化要高精度？**

A：归一化分母涉及均值/方差（或 RMS）累积，低精度会放大舍入误差并影响训练稳定性。

**Q(c)：混合精度速度收益？**

A：在 A100 上通常有明显收益，尤其 GEMM 密集模型；具体倍数依赖 batch、seq_len、激活重算与通信开销。

### 1.1.6 Profiling Memory

**Q(a)：显存峰值通常出现在哪？**

A：常见在 backward 末段或 `optimizer.step()` 前后（参数梯度 + optimizer state + 临时 buffer 叠加）。

**Q(b)：混合精度一定降显存吗？**

A：不一定。若引入额外 master weights/格式转换缓存，某些阶段峰值可能不降反升。

**Q(c)：attention 显存复杂度？**

A：标准 attention 需要显式 `N×N` score/prob，序列长度增大时内存与计算都会快速增长。

---

## 1.2 Attention

### 1.2.1 Benchmarking PyTorch Attention

**Q：长序列为什么容易 OOM？**

A：标准实现中 `S` 和 `P` 是 `O(N^2)` 张量，N 大时很快耗尽显存。

### 1.2.2 Optimizing Attention with FlashAttention-2

**Q：FlashAttention-2 的关键优化点？**

A：
- tile 化分块计算，避免显式存完整 `N×N`；
- online softmax（`m/l` 递推）提升数值稳定；
- 减少 HBM 往返，提升算子融合与访存效率。

### 1.2.3 FlashAttention-2 Forward Pass

**Q：为什么要保存 `L=logsumexp(S)`？**

A：backward 重算概率时使用 `P=exp(S-L)`，既稳定又避免保存完整 `P`。

### 1.2.4 OPTIONAL: Triton backward pass

**Q：Triton backward 如何拆分？**

A：可拆为三条路径：
- `dV = P^T @ dO`
- `dQ = (dS @ K) * scale`
- `dK = (dS^T @ Q) * scale`
其中 `dS = P * (dP - D)`，`D = sum(O*dO)`。

### 1.2.5 Benchmarking JIT-Compiled Attention

**Q：`torch.compile` 的预期收益？**

A：通常能减少 Python 调度开销并融合部分算子，forward/backward 都可能提速；收益依赖图稳定性与 dynamic shape 程度。

---

## 1.3 Distributed Data Parallel Training

### 1.3.1 Single-Node Distributed Communication in PyTorch

**Q：本作业核心通信原语？**

A：`broadcast`（参数同步）、`all_reduce`（梯度归约）、必要时 `all_gather/reduce_scatter`（高级优化）。

### 1.3.2 Naïve DDP

**Q：最小可行 DDP 做什么？**

A：
1. 初始化从 rank0 广播参数；
2. 每步 backward 后对梯度 all-reduce 并平均；
3. 各 rank 用一致梯度做本地 optimizer.step。

### 1.3.3 Overlap with individual gradients

**Q：如何实现通信-计算重叠？**

A：给参数注册梯度 hook，梯度一就绪即异步 all-reduce；反向结束后统一 wait。

### 1.3.4 Bucketed gradients

**Q：为什么 bucketed 通常更快？**

A：减少通信调用次数、提高单次消息大小，降低小包开销，更易与反向阶段重叠。

### 1.3.5 Improving minimal DDP

**Q：工程优化点有哪些？**

A：
- 真实 bucket flatten/反展平；
- 梯度视图复用减少内存复制；
- 更细粒度 stream/event 编排。

---

## 1.4 Optimizer State Sharding

**Q：状态分片为什么省显存？**

A：以 AdamW 为例，状态通常约是参数量的 2 倍（m、v）。按 rank 分片后，每个 rank 仅保留自己 shard 的状态，显存压力显著下降。

**Q：如何保证与非分片训练数学一致？**

A：
1. 先 all-reduce 梯度并平均；
2. owner rank 更新本 shard；
3. broadcast 更新后的参数到所有 rank。

---

## 1.5 4D Parallelism / Epilogue（概念题）

**Q：4D 并行的核心思想？**

A：将 Data / Tensor / Pipeline / Sequence(或 Context) 并行组合，按模型规模和互联带宽折中吞吐、延迟与显存。

**Q：系统实践中的总原则？**

A：先做 correctness，再做 profiling，再做针对性优化（通信重叠、kernel 融合、内存路径优化）。

---

## 本仓库当前实现对应

- FlashAttention PyTorch：已实现前后向与 causal。
- FlashAttention Triton：已实现 forward Triton kernel + backward Triton kernels。
- DDP：已实现参数广播与梯度同步（individual + bucketed API）。
- Sharded Optimizer：已实现索引分片 owner 更新 + 参数广播。

更详细的代码设计请见：`answer/implementation_details_zh.md`。
