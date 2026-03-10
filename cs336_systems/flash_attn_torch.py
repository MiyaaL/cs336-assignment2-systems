import torch
from einops import einsum, rearrange


def _apply_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    n_queries = scores.shape[-2]
    n_keys = scores.shape[-1]
    q_idx = torch.arange(n_queries, device=scores.device)[:, None]
    k_idx = torch.arange(n_keys, device=scores.device)[None, :]
    causal = q_idx >= k_idx
    return torch.where(causal, scores, torch.full_like(scores, -1e6))


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
    d_q = q.shape[-1]
    scale = 1 / (d_q ** 0.5)
    s = einsum(q, k, "b i d, b j d -> b i j") * scale
    if is_causal:
        s = _apply_causal_mask(s)
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
        if q.dim() < 3 or k.dim() < 3 or v.dim() < 3:
            raise ValueError("q, k, and v must have shape (Batch, ..., D)")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must have the same shape")

        merged_dims = q.shape[1:-1]
        q2 = rearrange(q, "b ... d -> b (...) d")
        k2 = rearrange(k, "b ... d -> b (...) d")
        v2 = rearrange(v, "b ... d -> b (...) d")

        d_q = q2.shape[-1]
        scale = 1 / (d_q ** 0.5)
        s = einsum(q2, k2, "b i d, b j d -> b i j") * scale
        if is_causal:
            s = _apply_causal_mask(s)

        p = torch.softmax(s, dim=-1)
        o2 = einsum(p, v2, "b i j, b j d -> b i d")
        l2 = torch.logsumexp(s, dim=-1)

        o = o2.view(q.shape)
        l = l2.view(q.shape[0], *merged_dims)

        ctx.save_for_backward(q2, k2, v2, o2, l2)
        ctx.is_causal = is_causal
        ctx.original_shape = q.shape
        return o

    @staticmethod
    def backward(ctx, do):
        q2, k2, v2, o2, l2 = ctx.saved_tensors
        do2 = rearrange(do, "b ... d -> b (...) d")
        D = torch.sum(o2 * do2, dim=-1)
        dq2, dk2, dv2 = flash_bwd_recompute_impl(q2, k2, v2, o2, do2, l2, D, ctx.is_causal)
        q_shape = ctx.original_shape
        dq = dq2.view(q_shape)
        dk = dk2.view(q_shape)
        dv = dv2.view(q_shape)
        return dq, dk, dv, None
