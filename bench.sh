#!?bin/bash

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# uv run nsys profile --stats=true -o nsys_prof/forward_small_model -t cuda,nvtx \
uv run python cs336_systems/benchmark.py \
    --mode full_training \
    --model-size xl \
    --device cuda \
    --dtype float16 \
    --memory-profiling \
    --warmup-steps 5 --num-steps 10 \
    --vocab-size 10000 \
    --batch-size 4 \
    --context-length 128 \
    --rope-theta 10000.0