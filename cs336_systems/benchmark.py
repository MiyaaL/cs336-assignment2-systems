import argparse
from timeit import default_timer as timer

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy

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
        choices=["forward", "forward_backward"],
        default="forward",
        help="Benchmark only forward pass or forward+backward.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the benchmark on.",
    )

    return parser.parse_args()


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
) -> float:
    use_cuda = device.type == "cuda"

    # Pre-create random targets for backward passes
    targets = None
    if mode == "forward_backward":
        targets = torch.randint(
            low=0,
            high=vocab_size,
            size=inputs.shape,
            device=device,
            dtype=torch.long,
        )

    # Warm-up
    for _ in range(warmup_steps):
        logits = model(inputs)
        if mode == "forward_backward":
            loss = cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1),
            )
            model.zero_grad(set_to_none=True)
            loss.backward()

        if use_cuda:
            torch.cuda.synchronize()

    # Timed steps
    start = timer()
    for _ in range(num_steps):
        logits = model(inputs)
        if mode == "forward_backward":
            loss = cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1),
            )
            model.zero_grad(set_to_none=True)
            loss.backward()

        if use_cuda:
            torch.cuda.synchronize()
    end = timer()

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
    )

    print(
        f"Mode: {args.mode}, device: {device.type}, "
        f"batch_size={args.batch_size}, seq_len={seq_len}, "
        f"avg_step_time={avg_step_time:.6f} seconds"
    )


if __name__ == "__main__":
    main()