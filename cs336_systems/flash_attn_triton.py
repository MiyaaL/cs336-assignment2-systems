import torch
import triton
import triton.language as tl


@triton.jit
def _flash_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    l_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    n_queries,
    n_keys,
    scale,
    d: tl.constexpr,
    q_tile_size: tl.constexpr,
    k_tile_size: tl.constexpr,
    is_causal: tl.constexpr,
):
    q_tile_idx = tl.program_id(0)
    b = tl.program_id(1)

    q_start = q_tile_idx * q_tile_size
    q_offsets = q_start + tl.arange(0, q_tile_size)
    d_offsets = tl.arange(0, d)

    q_ptrs = q_ptr + b * stride_qb + q_offsets[:, None] * stride_qq + d_offsets[None, :] * stride_qd
    q_mask = q_offsets[:, None] < n_queries
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    m_prev = tl.full((q_tile_size,), -float("inf"), dtype=tl.float32)
    l_prev = tl.zeros((q_tile_size,), dtype=tl.float32)
    o_acc = tl.zeros((q_tile_size, d), dtype=tl.float32)

    for k_start in range(0, n_keys, k_tile_size):
        k_offsets = k_start + tl.arange(0, k_tile_size)

        k_ptrs = k_ptr + b * stride_kb + k_offsets[:, None] * stride_kk + d_offsets[None, :] * stride_kd
        v_ptrs = v_ptr + b * stride_vb + k_offsets[:, None] * stride_vk + d_offsets[None, :] * stride_vd

        kv_mask = k_offsets[:, None] < n_keys
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        s = tl.dot(q, tl.trans(k)) * scale
        valid_k = k_offsets[None, :] < n_keys
        s = tl.where(valid_k, s, -float("inf"))

        if is_causal:
            causal_mask = q_offsets[:, None] >= k_offsets[None, :]
            s = tl.where(causal_mask, s, -float("inf"))

        m = tl.maximum(m_prev, tl.max(s, axis=1))
        p = tl.exp(s - m[:, None])
        alpha = tl.exp(m_prev - m)
        l = alpha * l_prev + tl.sum(p, axis=1)
        o_acc = alpha[:, None] * o_acc + tl.dot(p, v)

        m_prev = m
        l_prev = l

    o = o_acc / l_prev[:, None]

    o_ptrs = o_ptr + b * stride_ob + q_offsets[:, None] * stride_oq + d_offsets[None, :] * stride_od
    o_mask = q_offsets[:, None] < n_queries
    tl.store(o_ptrs, o, mask=o_mask)

    l_ptrs = l_ptr + b * stride_lb + q_offsets * stride_lq
    l_mask = q_offsets < n_queries
    tl.store(l_ptrs, m_prev + tl.log(l_prev), mask=l_mask)


@triton.jit
def _flash_bwd_dv_kernel(
    q_ptr,
    k_ptr,
    do_ptr,
    l_ptr,
    dv_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_lb,
    stride_lq,
    stride_dvb,
    stride_dvk,
    stride_dvd,
    n_queries,
    n_keys,
    scale,
    d: tl.constexpr,
    q_tile_size: tl.constexpr,
    k_tile_size: tl.constexpr,
    is_causal: tl.constexpr,
):
    k_tile_idx = tl.program_id(0)
    b = tl.program_id(1)

    k_start = k_tile_idx * k_tile_size
    k_offsets = k_start + tl.arange(0, k_tile_size)
    d_offsets = tl.arange(0, d)

    k_ptrs = k_ptr + b * stride_kb + k_offsets[:, None] * stride_kk + d_offsets[None, :] * stride_kd
    k_mask = k_offsets[:, None] < n_keys
    k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

    dv_acc = tl.zeros((k_tile_size, d), dtype=tl.float32)

    for q_start in range(0, n_queries, q_tile_size):
        q_offsets = q_start + tl.arange(0, q_tile_size)

        q_ptrs = q_ptr + b * stride_qb + q_offsets[:, None] * stride_qq + d_offsets[None, :] * stride_qd
        do_ptrs = do_ptr + b * stride_dob + q_offsets[:, None] * stride_doq + d_offsets[None, :] * stride_dod
        l_ptrs = l_ptr + b * stride_lb + q_offsets * stride_lq

        q_mask = q_offsets[:, None] < n_queries
        q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
        do_tile = tl.load(do_ptrs, mask=q_mask, other=0.0).to(tl.float32)
        l_tile = tl.load(l_ptrs, mask=q_offsets < n_queries, other=0.0).to(tl.float32)

        s = tl.dot(q_tile, tl.trans(k_tile)) * scale
        valid_q = q_offsets[:, None] < n_queries
        valid_k = k_offsets[None, :] < n_keys
        valid = valid_q & valid_k
        s = tl.where(valid, s, -float("inf"))

        if is_causal:
            causal_mask = q_offsets[:, None] >= k_offsets[None, :]
            s = tl.where(causal_mask, s, -float("inf"))

        p = tl.exp(s - l_tile[:, None])
        p = tl.where(valid, p, 0.0)
        dv_acc += tl.dot(tl.trans(p), do_tile)

    dv_ptrs = dv_ptr + b * stride_dvb + k_offsets[:, None] * stride_dvk + d_offsets[None, :] * stride_dvd
    dv_mask = k_offsets[:, None] < n_keys
    tl.store(dv_ptrs, dv_acc, mask=dv_mask)


@triton.jit
def _flash_bwd_dq_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    l_ptr,
    d_row_ptr,
    dq_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_lb,
    stride_lq,
    stride_db,
    stride_dq,
    stride_dqb,
    stride_dqq,
    stride_dqd,
    n_queries,
    n_keys,
    scale,
    d: tl.constexpr,
    q_tile_size: tl.constexpr,
    k_tile_size: tl.constexpr,
    is_causal: tl.constexpr,
):
    q_tile_idx = tl.program_id(0)
    b = tl.program_id(1)

    q_start = q_tile_idx * q_tile_size
    q_offsets = q_start + tl.arange(0, q_tile_size)
    d_offsets = tl.arange(0, d)

    q_ptrs = q_ptr + b * stride_qb + q_offsets[:, None] * stride_qq + d_offsets[None, :] * stride_qd
    do_ptrs = do_ptr + b * stride_dob + q_offsets[:, None] * stride_doq + d_offsets[None, :] * stride_dod
    l_ptrs = l_ptr + b * stride_lb + q_offsets * stride_lq
    drow_ptrs = d_row_ptr + b * stride_db + q_offsets * stride_dq

    q_tile = tl.load(q_ptrs, mask=q_offsets[:, None] < n_queries, other=0.0).to(tl.float32)
    do_tile = tl.load(do_ptrs, mask=q_offsets[:, None] < n_queries, other=0.0).to(tl.float32)
    l_tile = tl.load(l_ptrs, mask=q_offsets < n_queries, other=0.0).to(tl.float32)
    drow_tile = tl.load(drow_ptrs, mask=q_offsets < n_queries, other=0.0).to(tl.float32)

    dq_acc = tl.zeros((q_tile_size, d), dtype=tl.float32)

    for k_start in range(0, n_keys, k_tile_size):
        k_offsets = k_start + tl.arange(0, k_tile_size)

        k_ptrs = k_ptr + b * stride_kb + k_offsets[:, None] * stride_kk + d_offsets[None, :] * stride_kd
        v_ptrs = v_ptr + b * stride_vb + k_offsets[:, None] * stride_vk + d_offsets[None, :] * stride_vd

        kv_mask = k_offsets[:, None] < n_keys
        k_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        v_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        s = tl.dot(q_tile, tl.trans(k_tile)) * scale
        dp = tl.dot(do_tile, tl.trans(v_tile))

        valid_q = q_offsets[:, None] < n_queries
        valid_k = k_offsets[None, :] < n_keys
        valid = valid_q & valid_k
        s = tl.where(valid, s, -float("inf"))

        if is_causal:
            causal_mask = q_offsets[:, None] >= k_offsets[None, :]
            s = tl.where(causal_mask, s, -float("inf"))

        p = tl.exp(s - l_tile[:, None])
        p = tl.where(valid, p, 0.0)
        ds = p * (dp - drow_tile[:, None])

        dq_acc += tl.dot(ds, k_tile)

    dq_acc = dq_acc * scale

    dq_ptrs = dq_ptr + b * stride_dqb + q_offsets[:, None] * stride_dqq + d_offsets[None, :] * stride_dqd
    dq_mask = q_offsets[:, None] < n_queries
    tl.store(dq_ptrs, dq_acc, mask=dq_mask)


@triton.jit
def _flash_bwd_dk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    l_ptr,
    d_row_ptr,
    dk_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_lb,
    stride_lq,
    stride_db,
    stride_dq,
    stride_dkb,
    stride_dkk,
    stride_dkd,
    n_queries,
    n_keys,
    scale,
    d: tl.constexpr,
    q_tile_size: tl.constexpr,
    k_tile_size: tl.constexpr,
    is_causal: tl.constexpr,
):
    k_tile_idx = tl.program_id(0)
    b = tl.program_id(1)

    k_start = k_tile_idx * k_tile_size
    k_offsets = k_start + tl.arange(0, k_tile_size)
    d_offsets = tl.arange(0, d)

    k_ptrs = k_ptr + b * stride_kb + k_offsets[:, None] * stride_kk + d_offsets[None, :] * stride_kd
    v_ptrs = v_ptr + b * stride_vb + k_offsets[:, None] * stride_vk + d_offsets[None, :] * stride_vd

    k_tile = tl.load(k_ptrs, mask=k_offsets[:, None] < n_keys, other=0.0).to(tl.float32)
    v_tile = tl.load(v_ptrs, mask=k_offsets[:, None] < n_keys, other=0.0).to(tl.float32)

    dk_acc = tl.zeros((k_tile_size, d), dtype=tl.float32)

    for q_start in range(0, n_queries, q_tile_size):
        q_offsets = q_start + tl.arange(0, q_tile_size)

        q_ptrs = q_ptr + b * stride_qb + q_offsets[:, None] * stride_qq + d_offsets[None, :] * stride_qd
        do_ptrs = do_ptr + b * stride_dob + q_offsets[:, None] * stride_doq + d_offsets[None, :] * stride_dod
        l_ptrs = l_ptr + b * stride_lb + q_offsets * stride_lq
        drow_ptrs = d_row_ptr + b * stride_db + q_offsets * stride_dq

        q_tile = tl.load(q_ptrs, mask=q_offsets[:, None] < n_queries, other=0.0).to(tl.float32)
        do_tile = tl.load(do_ptrs, mask=q_offsets[:, None] < n_queries, other=0.0).to(tl.float32)
        l_tile = tl.load(l_ptrs, mask=q_offsets < n_queries, other=0.0).to(tl.float32)
        drow_tile = tl.load(drow_ptrs, mask=q_offsets < n_queries, other=0.0).to(tl.float32)

        s = tl.dot(q_tile, tl.trans(k_tile)) * scale
        dp = tl.dot(do_tile, tl.trans(v_tile))

        valid_q = q_offsets[:, None] < n_queries
        valid_k = k_offsets[None, :] < n_keys
        valid = valid_q & valid_k
        s = tl.where(valid, s, -float("inf"))

        if is_causal:
            causal_mask = q_offsets[:, None] >= k_offsets[None, :]
            s = tl.where(causal_mask, s, -float("inf"))

        p = tl.exp(s - l_tile[:, None])
        p = tl.where(valid, p, 0.0)
        ds = p * (dp - drow_tile[:, None])

        dk_acc += tl.dot(tl.trans(ds), q_tile)

    dk_acc = dk_acc * scale

    dk_ptrs = dk_ptr + b * stride_dkb + k_offsets[:, None] * stride_dkk + d_offsets[None, :] * stride_dkd
    dk_mask = k_offsets[:, None] < n_keys
    tl.store(dk_ptrs, dk_acc, mask=dk_mask)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            raise ValueError("q, k, and v must have shape (Batch, SeqLen, D)")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must have the same shape")
        if q.device.type != "cuda" or k.device.type != "cuda" or v.device.type != "cuda":
            raise ValueError("Input tensors must be on CUDA")

        bsz, n_queries, d = q.shape
        _, n_keys, _ = k.shape

        o = torch.empty_like(q)
        l = torch.empty((bsz, n_queries), device=q.device, dtype=torch.float32)

        q_tile_size = 16
        k_tile_size = 16
        grid = (triton.cdiv(n_queries, q_tile_size), bsz)

        _flash_fwd_kernel[grid](
            q,
            k,
            v,
            o,
            l,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            l.stride(0),
            l.stride(1),
            n_queries,
            n_keys,
            1.0 / (d ** 0.5),
            d=d,
            q_tile_size=q_tile_size,
            k_tile_size=k_tile_size,
            is_causal=is_causal,
        )

        ctx.save_for_backward(q, k, v, o, l)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l = ctx.saved_tensors
        bsz, n_queries, d = q.shape
        _, n_keys, _ = k.shape
        scale = 1.0 / (d ** 0.5)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        d_row = torch.sum(o * do, dim=-1).to(torch.float32)

        q_tile_size = 16
        k_tile_size = 16

        dv_grid = (triton.cdiv(n_keys, k_tile_size), bsz)
        _flash_bwd_dv_kernel[dv_grid](
            q,
            k,
            do,
            l,
            dv,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            l.stride(0),
            l.stride(1),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            n_queries,
            n_keys,
            scale,
            d=d,
            q_tile_size=q_tile_size,
            k_tile_size=k_tile_size,
            is_causal=ctx.is_causal,
        )

        dq_grid = (triton.cdiv(n_queries, q_tile_size), bsz)
        _flash_bwd_dq_kernel[dq_grid](
            q,
            k,
            v,
            do,
            l,
            d_row,
            dq,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            l.stride(0),
            l.stride(1),
            d_row.stride(0),
            d_row.stride(1),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            n_queries,
            n_keys,
            scale,
            d=d,
            q_tile_size=q_tile_size,
            k_tile_size=k_tile_size,
            is_causal=ctx.is_causal,
        )

        dk_grid = (triton.cdiv(n_keys, k_tile_size), bsz)
        _flash_bwd_dk_kernel[dk_grid](
            q,
            k,
            v,
            do,
            l,
            d_row,
            dk,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            l.stride(0),
            l.stride(1),
            d_row.stride(0),
            d_row.stride(1),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            n_queries,
            n_keys,
            scale,
            d=d,
            q_tile_size=q_tile_size,
            k_tile_size=k_tile_size,
            is_causal=ctx.is_causal,
        )

        return dq, dk, dv, None
