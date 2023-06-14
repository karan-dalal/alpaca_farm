current_t=$1
dump_directory=$2
counter=$3

python3 -m torch.distributed.run --nproc_per_node=8 --master_port=1234 refit_model.py \
  --fp16 False \
  --bf16 False \
  --seed 42 \
  --model_max_length 512 \
  --num_train_epochs 6 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 1000000000 \
  --save_total_limit 1 \
  --learning_rate 3e-6 \
  --weight_decay 0.0 \
  --logging_steps 10 \
  --logging_strategy epoch \
  --wandb_project "alpaca_farm" \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
  --tf32 False \
  --flash_attn True \
  --run_name "t=${current_t}" \
  --ddp_timeout 1800 \
  --current_t $current_t \
  --output_dir "${dump_directory}" \
  --counter $counter \
  --do_eval False \
  --logging_dir "${dump_directory}/t=${current_t}"