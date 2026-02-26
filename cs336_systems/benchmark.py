import argparse
from timeit import default_timer as timer
from contextlib import nullcontext

import torch.cuda.nvtx as nvtx
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark BasicsTransformerLM forward and backward passes."
    )

    # Model hyperparameters
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    # Data / batch
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length for the random batch (defaults to context-length).",
    )

    # Benchmark settings
    parser.add_argument(
        "--warmup-steps",
        "-w",
        type=int,
        default=5,
        help="Number of warm-up steps before timing.",
    )
    parser.add_argument(
        "--num-steps",
        "-n",
        type=int,
        default=10,
        help="Number of timed steps.",
    )
    parser.add_argument(
        "--mode",
        choices=["forward", "forward_backward", "full_training"],
        default="forward",
        help="Benchmark only forward pass or forward+backward.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the benchmark on.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Data type to use for the benchmark.",
    )
    parser.add_argument(
        "--model-size",
        choices=["small", "medium", "large", "xl", "2.7b"],
        default=None,
        help="Model size to use for the benchmark.",
    )

    # Memory benchmark
    parser.add_argument(
        "--memory-profiling",
        action="store_true",
        help="Enable memory profiling.",
    )

    args = parser.parse_args()

    MODEL_SIZES = {
        "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
        "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
        "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
        "xl": dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
        "2.7b": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    }

    if args.model_size is not None:
        cfg = MODEL_SIZES.get(args.model_size)
        if cfg is None:
            raise ValueError(f"Invalid model size: {args.model_size}")
        args.d_model = cfg["d_model"]
        args.d_ff = cfg["d_ff"]
        args.num_layers = cfg["num_layers"]
        args.num_heads = cfg["num_heads"]

    return args


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_str]


def create_model(args: argparse.Namespace, device: torch.device) -> BasicsTransformerLM:
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    model.to(device)
    return model


def create_random_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    # Tokens in [0, vocab_size)
    return torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )


def benchmark(
    model: BasicsTransformerLM,
    inputs: torch.Tensor,
    vocab_size: int,
    warmup_steps: int,
    num_steps: int,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    model_size: str | None,
    memory_profiling: bool = False,
) -> float:
    use_cuda = device.type == "cuda"

    # Pre-create random targets for backward passes
    targets = None
    if mode != "forward":
        targets = torch.randint(
            low=0,
            high=vocab_size,
            size=inputs.shape,
            device=device,
            dtype=torch.long,
        )
    optimizer = AdamW(model.parameters(), lr=1e-3)

    cast_context = torch.autocast(device_type="cuda", dtype=dtype) if dtype != torch.float32 else nullcontext()

    with cast_context:
        # Warm-up
        for _ in range(warmup_steps):
            if mode == "forward":
                with torch.no_grad():
                    logits = model(inputs)
            elif mode == "forward_backward":
                model.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = cross_entropy(
                    logits.view(-1, vocab_size),
                    targets.view(-1),
                )
                loss.backward()
            elif mode == "full_training":
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = cross_entropy(
                    logits.view(-1, vocab_size),
                    targets.view(-1),
                )
                loss.backward()
                optimizer.step()
            else:
                raise ValueError(f"Invalid mode: {mode}")

            if use_cuda:
                torch.cuda.synchronize()
        
        if memory_profiling:
            torch.cuda.memory._record_memory_history(max_entries=1000000)

        # Timed steps
        start = timer()
        for _ in range(num_steps):
            if mode == "forward":
                with nvtx.range("forward"):
                    with torch.no_grad():
                        logits = model(inputs)
            elif mode == "forward_backward":
                with nvtx.range("forward_backward"):
                    model.zero_grad(set_to_none=True)
                    logits = model(inputs)
                    loss = cross_entropy(
                        logits.view(-1, vocab_size),
                        targets.view(-1),
                    )
                    loss.backward()
            elif mode == "full_training":
                with nvtx.range("full_training"):
                    optimizer.zero_grad(set_to_none=True)
                    with nvtx.range("forward"):
                        logits = model(inputs)
                    with nvtx.range("loss"):
                        loss = cross_entropy(
                            logits.view(-1, vocab_size),
                            targets.view(-1),
                        )
                    with nvtx.range("backward"):
                        loss.backward()
                    with nvtx.range("step"):
                        optimizer.step()
            else:
                raise ValueError(f"Invalid mode: {mode}")

            if use_cuda:
                torch.cuda.synchronize()
        end = timer()

        if memory_profiling:
            if model_size is not None:
                memory_path = "./mem_prof/memory_" + model_size + "_"  + "ctx-len" + str(inputs.shape[1]) + "_" + str(dtype) + "_" + mode + ".pickle"
            else:
                memory_path = "./mem_prof/memory_" + "ctx-len" + str(inputs.shape[1]) + "_" + str(dtype) + "_" + mode + ".pickle"
            torch.cuda.memory._dump_snapshot(memory_path)
            torch.cuda.memory._record_memory_history(enabled=None)
        
        total_time = end - start

    return total_time / num_steps


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but device='cuda' was requested.")

    seq_len = args.seq_len or args.context_length

    model = create_model(args, device)
    inputs = create_random_batch(
        batch_size=args.batch_size,
        seq_len=seq_len,
        vocab_size=args.vocab_size,
        device=device,
    )

    avg_step_time = benchmark(
        model=model,
        inputs=inputs,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        num_steps=args.num_steps,
        mode=args.mode,
        device=device,
        dtype=get_torch_dtype(args.dtype),
        model_size=args.model_size,
        memory_profiling=args.memory_profiling,
    )

    print(
        f"Mode: {args.mode}, device: {device.type}, "
        f"batch_size={args.batch_size}, seq_len={seq_len}, "
        f"avg_step_time={avg_step_time:.6f} seconds"
    )


if __name__ == "__main__":
    main()