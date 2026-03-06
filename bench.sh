#!?bin/bash

# 1.1
# uv run nsys profile --stats=true -t cuda,nvtx \
#     python cs336_systems/benchmark.py \
#     --mode full_training \
#     --model-size xl \
#     --device cuda \
#     --dtype float16 \
#     --memory-profiling \
#     --warmup-steps 5 --num-steps 10 \
#     --vocab-size 10000 \
#     --batch-size 4 \
#     --context-length 128 \
#     --rope-theta 10000.0

# 1.2
d_model=(16 32 64 128)
seq_len=(256 1024 4096 8192)

for d in ${d_model[@]}; do
    for l in ${seq_len[@]}; do
        uv run nsys profile --stats=true -t cuda,nvtx \
            python cs336_systems/bench_attention.py \
            --d-model $d --seq-len $l \
            --func torch \
            --mem-prof \
            --prof-mode full 2>&1 | tee -a ./logs/attention_bench_${d}_${l}_full.log
    done
done