export CUDA_VISIBLE_DEVICES=0

DATASET="tabfact" 

# MODEL_TYPE="gpt4o"
MODEL_TYPE="llama3.2_3b"
# MODEL_TYPE="gemma2_27b"
# MODEL_TYPE="qwen2.5_3b"

python3 tweval.py \
    --model ${MODEL_TYPE} \
    --adapter_path example_ckpt/tw_evaluator_ckpt \
    --max_count 100000 \
    --start_idx 0 \
    --output ../outputs/${DATASET}_${MODEL_TYPE}_twe_result.jsonl \
    --dataset ${DATASET} 
