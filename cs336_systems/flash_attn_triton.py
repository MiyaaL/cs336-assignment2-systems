import torch
import triton
import triton.language as tl

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_mq, stride_mk,
    N_QUERIES,
    N_KEYS,
    scale: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    mask,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0, ),
    )
    if is_causal:
        mask_block_ptr = tl.make_block_ptr(
            mask,
            shape=(N_QUERIES, N_KEYS),
            strides=(stride_mq, stride_mk),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, K_TILE_SIZE),
            order=(1, 0),
        )

    Q_i_raw = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Q_i = Q_i_raw.to(tl.float32) # softmax in fp32 to prevent overflow

    K_j_ptr = K_block_ptr
    V_j_ptr = V_block_ptr

    m_prev = tl.full((Q_TILE_SIZE, ), -float("inf"), dtype=Q_i.dtype)
    l = tl.zeros((Q_TILE_SIZE, ), dtype=Q_i.dtype)
    o = tl.zeros((Q_TILE_SIZE, D), dtype=Q_i.dtype)
    for _ in range(0, N_KEYS, K_TILE_SIZE):
        K_j = tl.load(K_j_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32) # softmax in fp32 to prevent overflow
        V_j = tl.load(V_j_ptr, boundary_check=(0, 1), padding_option="zero")
        S = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            mask_j = tl.load(mask_block_ptr, boundary_check=(0, 1), padding_option="zero")
            S += mask_j
            mask_block_ptr = tl.advance(mask_block_ptr, (0, K_TILE_SIZE))
        m = tl.maximum(m_prev, tl.max(S, axis=1))
        P = tl.exp(S - m[:, None])
        alpha = tl.exp(m_prev - m)
        l = alpha * l + tl.sum(P, axis=1)
        P = P.to(Q_i_raw.dtype)
        o = tl.dot(P, V_j, acc=alpha[:, None]*o)
        m_prev = m
        K_j_ptr = tl.advance(K_j_ptr, (K_TILE_SIZE, 0))
        V_j_ptr = tl.advance(V_j_ptr, (K_TILE_SIZE, 0))

    tl.store(O_block_ptr, (o / l[:, None]).to(Q_i_raw.dtype), boundary_check=(0, 1))
    tl.store(L_block_ptr, m_prev + tl.log(l), boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        if (q.dim() != 3 or k.dim() != 3 or v.dim() != 3):
            raise ValueError("q, k, and v must have shape (Batch, SeqLen, D)")
        if (q.shape != k.shape or q.shape != v.shape):
            raise ValueError("q, k, and v must have the same shape")
        assert (q.device.type == "cuda" and
                k.device.type == "cuda" and
                v.device.type == "cuda"), "Input tensors must be on CUDA and is_causal must be False"
        
        # shape
        B, Q, D = q.shape
        _, K, _ = k.shape
        _, V, _ = v.shape

        # mask
        if is_causal:
            mask = torch.ones(Q, K, device=q.device, dtype=q.dtype) * -1e6
            mask = mask.triu(diagonal=1)
        else:
            mask = None

        # output and loss tensors
        O = torch.empty(B, Q, D, device=q.device, dtype=q.dtype)
        L = torch.zeros(B, Q, device=q.device, dtype=torch.float32)

        # strides
        stride_qb = q.stride(0)
        stride_qq = q.stride(1)
        stride_qd = q.stride(2)
        stride_kb = k.stride(0)
        stride_kk = k.stride(1)
        stride_kd = k.stride(2)
        stride_vb = v.stride(0)
        stride_vk = v.stride(1)
        stride_vd = v.stride(2)
        stride_ob = O.stride(0)
        stride_oq = O.stride(1)
        stride_od = O.stride(2)
        stride_lb = L.stride(0)
        stride_lq = L.stride(1)
        stride_mq = mask.stride(0) if is_causal else None
        stride_mk = mask.stride(1) if is_causal else None
        
        # tile size
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        grid = (triton.cdiv(Q, Q_TILE_SIZE), B)

        # launch kernel
        flash_fwd_kernel[grid](
            q, k, v,
            O, L,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            stride_mq, stride_mk,
            N_QUERIES=Q, N_KEYS=K,
            scale=1.0 / (D ** 0.5),
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
            mask=mask,
        )

        ctx.save_for_backward(q, k, v, O, L, mask)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx):
        q, k, v, O, L, mask = ctx.saved_tensors
        B, Q, D = q.shape
        _, K, _ = k.shape
        _, V, _ = v.shape

        # output and loss tensors
        dq = torch.empty(B, Q, D, device=q.device, dtype=q.dtype)
        dk = torch.empty(B, K, D, device=k.device, dtype=k.dtype)
        dv = torch.empty(B, K, D, device=v.device, dtype=v.dtype)
        return dq, dk, dv, None, None
        