export CUDA_VISIBLE_DEVICES=0

DATASET="tabfact"

# MODEL="google/gemma-2-2b-it"
# MODEL="Qwen/Qwen2.5-32B-Instruct"
MODEL="meta-llama/llama-3.2-3b-instruct"
# MODEL="gpt-4o"
# MODEL_TYPE="gpt4o"
MODEL_TYPE="llama3.2_3b"
# MODEL_TYPE="gemma2_2b"
# MODEL_TYPE="qwen2.5_32b"

python3 run.py \
    --model_name "${MODEL}" \
    --adapter_path example_ckpt/ans_selector_ckpt \
    --max_count 100000 \
    --start_idx 0 \
    --agent ../mixsc/output/${DATASET}_test_agent_${MODEL_TYPE}/result.jsonl \
    --dp ../mixsc/output/${DATASET}_test_dp_${MODEL_TYPE}/result.jsonl \
    --output outputs/${DATASET}_${MODEL_TYPE}_result.jsonl \
    --dataset ${DATASET} 
