# CS336 Assignment2 Systems 实现细节（中文）

本文档说明 `cs336_systems/` 中的核心实现思路、关键公式、以及与测试脚本的对应关系。

---

## 1. FlashAttention（PyTorch 版）

对应文件：`cs336_systems/flash_attn_torch.py`

### 1.1 Forward

- 输入允许 `(B, ..., D)`，先通过 `einops.rearrange(q, "b ... d -> b (...) d")` 合并中间维度，统一成 `(B, N, D)`。
- 计算打分矩阵：
  \[
  S = QK^T / \sqrt{D}
  \]
- 如果 `is_causal=True`，对未来位置做 mask（未来位置设为 `-1e6`）。
- 概率矩阵：
  \[
  P = \operatorname{softmax}(S)
  \]
- 输出：
  \[
  O = PV
  \]
- 额外保存
  \[
  L = \log\sum\exp(S)
  \]
  用于 backward 重算时的数值稳定。

### 1.2 Backward

- 接收上游梯度 `dO`，重排成 `(B, N, D)`。
- 通过 `L` 重算
  \[
  P = \exp(S - L)
  \]
- 使用标准 FlashAttention-2 推导形式：
  - \(D_i = \sum_d O_{i,d} \cdot dO_{i,d}\)
  - \(dV = P^T dO\)
  - \(dP = dO V^T\)
  - \(dS = P \odot (dP - D)\)
  - \(dQ = dS K / \sqrt{D}\)
  - \(dK = dS^T Q / \sqrt{D}\)

### 1.3 与测试约束的对应

- 测试会检查 saved tensor 中是否存在一个形状为 `(B, N)` 的张量（即 `L`），该实现满足。
- 前向/反向都支持 `is_causal=False/True`。

---

## 2. FlashAttention（Triton 版）

对应文件：`cs336_systems/flash_attn_triton.py`

### 2.1 Triton Forward Kernel

- `_flash_fwd_kernel` 以 `(query_tile, batch)` 为 grid。
- 使用 online softmax 递推（`m_prev/l_prev`）避免数值溢出：
  - `m = max(m_prev, max(S_j))`
  - `p = exp(S_j - m)`
  - `alpha = exp(m_prev - m)`
  - `l = alpha * l_prev + sum(p)`
  - `o = alpha * o_prev + p @ V_j`
- 最终写出 `O = o / l` 和 `L = m + log(l)`。
- 同时处理：
  - 非整除 tile 的边界 mask
  - `is_causal` 的三角 mask

### 2.2 Triton Backward Kernels

目前 backward 已使用 Triton 内核实现三条梯度路径：

1. `_flash_bwd_dv_kernel`
   - 对每个 key-tile 累加 `dv = P^T @ dO`
2. `_flash_bwd_dq_kernel`
   - 计算 `dP = dO @ V^T`
   - 用 `L` 和 `D_row=sum(O*dO)` 重建 `dS = P*(dP-D_row)`
   - 累加 `dQ = (dS @ K) * scale`
3. `_flash_bwd_dk_kernel`
   - 同理计算 `dK = (dS^T @ Q) * scale`

### 2.3 数值与精度

- kernel 内部主累加使用 `fp32`。
- 输出梯度写回与输入同 dtype。

---

## 3. DDP（individual parameter）

对应文件：`cs336_systems/distributed.py` 中 `DDPIndividualParameters`

### 3.1 初始化同步

- 构造时从 rank0 广播参数与 buffer，保证各 rank 初始模型一致。

### 3.2 反向通信

- 给每个 `requires_grad=True` 参数注册 hook。
- 每个参数梯度就绪后触发 `dist.all_reduce(..., async_op=True)`。
- 在 `finish_gradient_synchronization()` 中统一 `wait()` 并除以 `world_size`。

此设计满足“参数级通信 + 与反向重叠”的最小可行实现。

---

## 4. DDP（bucketed）

对应文件：`cs336_systems/distributed.py` 中 `DDPBucketed`

- 保留 bucketed DDP 对外接口（含 `bucket_size_mb` 与 `on_train_batch_start`）。
- 当前实现为 correctness-first 路线：复用参数级 hook 同步策略，先保证数值正确和测试兼容。
- 若追求性能，可进一步实现真实 bucket flatten + bucket-ready all-reduce。

---

## 5. Sharded Optimizer

对应文件：`cs336_systems/distributed.py` 中 `ShardedOptimizer`

### 5.1 分片策略

- 按参数索引 `idx % world_size == rank` 划分 owner。
- 每个 rank 本地仅维护 owner 参数的 optimizer state。

### 5.2 训练步骤

1. 所有参数梯度先 all-reduce 并平均（保证数学等价于非分片全局梯度）。
2. 本地 optimizer 仅更新 owner shard。
3. owner rank 将更新后的参数 broadcast 给所有 rank。

该流程可在保证收敛行为一致的同时，减少每个 rank 的 optimizer state 占用。

---

## 6. adapters 接线

对应文件：`tests/adapters.py`

已将测试入口全部接线到实际实现：

- `get_flashattention_autograd_function_pytorch` → `FlashAttentionTorch`
- `get_flashattention_autograd_function_triton` → `FlashAttentionTriton`
- `get_ddp_individual_parameters` → `DDPIndividualParameters`
- `ddp_individual_parameters_on_after_backward` → `finish_gradient_synchronization`
- `get_ddp_bucketed` → `DDPBucketed`
- `ddp_bucketed_on_after_backward` → `finish_gradient_synchronization`
- `ddp_bucketed_on_train_batch_start` → `on_train_batch_start`
- `get_sharded_optimizer` → `ShardedOptimizer`

---

## 7. 与你运行环境（ARM + CentOS + 2×A100-40G）相关建议

1. Triton/PyTorch 版本建议固定并与 CUDA 驱动严格匹配（避免 ABI 不兼容）。
2. attention kernel 的 tile size 可在 A100 上进一步调参（例如 32/64）。
3. 若目标是吞吐，建议下一步实现：
   - bucket 真正扁平化通信
   - backward 进一步减少重复访存
   - `torch.compile` 与 Triton kernel 联调 benchmark。
