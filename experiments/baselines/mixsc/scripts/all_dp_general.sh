# This script runs dp on all wtq datasets using gpt-3.5
# - tables are not perturbed
# - resorting stage in NORM is disabled

DATASET="penguin"

# CUDA_VISIBLE_DEVICES=0 python run_cot_general.py \
#     --model google/gemma-2-2b-it \
#     --provider vllm --dataset "$DATASET" --split train \
#     --perturbation none --norm True --disable_resort True --norm_cache True \
#     --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.1 \
#     --log_dir output/${DATASET}_test_dp_gemma2_2b --cache_dir cache/${DATASET}_test_dp_gemma2_2b

# CUDA_VISIBLE_DEVICES=0 python run_cot_general.py \
#     --model Qwen/Qwen2.5-7B-Instruct \
#     --provider vllm --dataset "$DATASET" --split train \
#     --perturbation none --norm True --disable_resort True --norm_cache True \
#     --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.1 \
#     --log_dir output/${DATASET}_test_dp_qwen2.5_7b --cache_dir cache/${DATASET}_test_dp_qwen2.5_7b

CUDA_VISIBLE_DEVICES=0 python run_cot_general.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --provider vllm --dataset "$DATASET" --split test \
    --perturbation none --norm True --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.1 \
    --log_dir output/${DATASET}_test_dp_llama3.2_3b --cache_dir cache/${DATASET}_test_dp_llama3.2_3b

# CUDA_VISIBLE_DEVICES=0 python run_cot_general.py \
#     --model gpt-4o \
#     --provider openai --dataset "$DATASET" --split train \
#     --perturbation none --norm True --disable_resort True --norm_cache True \
#     --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.1 \
#     --log_dir output/${DATASET}_test_dp_gpt4o --cache_dir cache/${DATASET}_test_dp_gpt4o


    # --model Qwen/Qwen2.5-Coder-14B-Instruct \
    # --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \