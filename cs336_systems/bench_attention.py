import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

import argparse
import tqdm
from contextlib import nullcontext

from cs336_basics.model import annotated_scaled_dot_product_attention, scaled_dot_product_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark scaled dot product attention."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--prof-mode", type=str, default="forward", choices=["forward", "backward", "full"])
    parser.add_argument("--mem-prof", action="store_true")
    parser.add_argument("--func", type=str, choices=["torch", "torch_compiled"], default="torch")
    return parser.parse_args()

def benchmark_attention(
    q, k, v,
    scaled_dot_product_attention_func,
    prof_mode : str = "forward",
    mem_prof : bool = False,
    device : torch.device = torch.device("cuda")
):
    # warm up
    for _ in range(10):
        o = scaled_dot_product_attention_func(q, k, v)
        torch.cuda.synchronize()
        loss = o.sum()
        torch.cuda.synchronize()
        loss.backward()
        torch.cuda.synchronize()
    # benchmark initial
    for _ in range(100):
        nvtx_ctx = nvtx.range("attention_forward") if prof_mode == "forward" or "full" else nullcontext()
        if mem_prof and prof_mode == "forward" or "full":
            torch.cuda.memory._record_memory_history(max_entries=100000)
        with nvtx_ctx:
            o = scaled_dot_product_attention_func(q, k, v)
            torch.cuda.synchronize()
        if mem_prof and prof_mode == "forward" or "full":
            mem_path = f"./mem_prof/torch_attn_memory_{q.shape[-1]}_{q.shape[-2]}_forward.pickle"
            torch.cuda.memory._dump_snapshot(mem_path)
            torch.cuda.memory._record_memory_history(enabled=None)
        loss = o.sum()
        torch.cuda.synchronize()

        nvtx_ctx = nvtx.range("attention_backward") if prof_mode == "backward" or "full" else nullcontext()
        if mem_prof and prof_mode == "backward" or "full":
            torch.cuda.memory._record_memory_history(max_entries=100000)
        with nvtx_ctx:
            loss.backward()
            torch.cuda.synchronize()
        if mem_prof and prof_mode == "backward" or "full":
            mem_path = f"./mem_prof/torch_attn_memory_{q.shape[-1]}_{q.shape[-2]}_backward.pickle"
            torch.cuda.memory._dump_snapshot(mem_path)
            torch.cuda.memory._record_memory_history(enabled=None)


def main():
    args = parse_args()

    batch_size = args.batch_size
    num_heads = args.num_heads
    d_model = [args.d_model]
    seq_len = [args.seq_len]
    prof_mode = args.prof_mode
    mem_prof = args.mem_prof
    device = torch.device("cuda")

    if args.func == "torch":
        func = scaled_dot_product_attention
    else:
        func = torch.compile(scaled_dot_product_attention)

    for d in tqdm.tqdm(d_model, desc="d_model"):
        for l in tqdm.tqdm(seq_len, desc="seq_len"):
            q = torch.randn(batch_size, num_heads, l, d, device=device, requires_grad=True)
            k = torch.randn(batch_size, num_heads, l, d, device=device, requires_grad=True)
            v = torch.randn(batch_size, num_heads, l, d, device=device, requires_grad=True)
            benchmark_attention(q, k, v, func, prof_mode=prof_mode, mem_prof=mem_prof, device=device)

if __name__ == "__main__":
    main()