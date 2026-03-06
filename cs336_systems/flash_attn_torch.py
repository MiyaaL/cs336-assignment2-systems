import torch
import math
from einops import einsum, rearrange


def flash_bwd_recompute_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    L: torch.Tensor,
    D: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # recompute P
    d_q = q.shape[-1]
    scale = 1 / (d_q ** 0.5)
    s = einsum(q, k, "b i d, b j d -> b i j") * scale
    p = torch.exp(s - L.unsqueeze(-1))

    dv = einsum(p, do, "b i j, b i d -> b j d")
    dp = einsum(do, v, "b i d, b j d -> b i j")
    ds = p * (dp - D.unsqueeze(-1))
    dq = einsum(ds, k, "b i j, b j d -> b i d") * scale
    dk = einsum(ds, q, "b i j, b i d -> b j d") * scale
    return dq, dk, dv


class FlashAttentionTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        if (q.dim() < 3 or k.dim() < 3 or v.dim() < 3):
            raise ValueError("q, k, and v must have shape (Batch, ..., D)")
        if (q.shape != k.shape or q.shape != v.shape):
            raise ValueError("q, k, and v must have the same shape")

        # tile size
        B_q = 16
        B_k = 16
        # output shape
        merged_dims = q.shape[1:-1]
        q = rearrange(q, "b ... d -> b (...) d")
        k = rearrange(k, "b ... d -> b (...) d")
        v = rearrange(v, "b ... d -> b (...) d")
        # input size
        batch_size, N_q, d_q = q.shape
        _, N_k, d_k = k.shape
        _, N_v, d_v = v.shape

        # output
        o = torch.zeros(batch_size, N_q, d_q, device=q.device)
        l = torch.zeros(batch_size, N_q, device=q.device)

        # get tile
        T_q = N_q // B_q
        T_k = N_k // B_k
       
        # for tiles in q
        for i in range(T_q):
            q_tile = q[:,i * B_q:(i + 1) * B_q, :]
            o_prev = torch.zeros(batch_size, B_q, d_q, dtype=q.dtype, device=q.device)
            m_prev = torch.ones(batch_size, B_q, dtype=q.dtype, device=q.device) * -float('inf')
            l_prev = torch.zeros(batch_size, B_q, dtype=q.dtype, device=q.device)
            for j in range(T_k):
                k_tile = k[:, j * B_k:(j + 1) * B_k, :]
                v_tile = v[:, j * B_k:(j + 1) * B_k, :]

                s_j = einsum(q_tile, k_tile, "b i d, b j d -> b i j") / (d_q ** 0.5)
                m_j = torch.maximum(m_prev, torch.max(s_j, dim=-1, keepdim=False)[0])
                p_j = torch.exp(s_j - m_j.unsqueeze(-1).repeat(1, 1, B_k))
                l_j = torch.exp(m_prev - m_j) * l_prev + torch.sum(p_j, dim=-1, keepdim=False)
                o_j = torch.exp(m_prev - m_j).unsqueeze(-1) * o_prev + einsum(p_j, v_tile, "b i j, b j d -> b i d")

                m_prev = m_j
                l_prev = l_j
                o_prev = o_j
        
            o_i = o_j / l_j.unsqueeze(-1)
            l_i = m_j + torch.log(l_j)
            # fill output
            o[:, i * B_q:(i + 1) * B_q, :] = o_i
            l[:, i * B_q:(i + 1) * B_q] = l_i
        
        l = l.view(batch_size, *merged_dims)
        o = o.view(batch_size, *merged_dims, d_q)

        ctx.save_for_backward(q, k, v, o, l)
        ctx.is_causal = is_causal

        return o

    @staticmethod
    def backward(ctx, do):
        (q, k, v, o, L) = ctx.saved_tensors
        D = torch.sum(o * do, dim=-1)
        dq, dk, dv = flash_bwd_recompute_impl(q, k, v, o, do, L, D, ctx.is_causal)
        return dq, dk, dv, None