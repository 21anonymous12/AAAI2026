export CUDA_VISIBLE_DEVICES=0

# model_name="neulab/omnitab-large"
model_name="google/gemma-2-27b-it"
# model_name="Qwen/Qwen2.5-32B-Instruct"
# model_name="meta-llama/Llama-3.2-3B-Instruct"

dataset_name="wikitq"
output_dir="output/wikitq_tableqa1"
checkpoint=4050

python ./run_vllm.py \
  --task tableqa \
  --do_predict \
  --squall_plus True \
  --predict_split test \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_name} \
  --max_source_length 4096 \
  --max_target_length 2048 \
  --val_max_target_length 2048 \
  --per_device_eval_batch_size 1 \
  --dataset_name ${dataset_name} \
  --split_id 2 \
  --predict_with_generate \
  --num_beams 1 \
  --max_predict_samples 100000

# --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \

# --aug True 