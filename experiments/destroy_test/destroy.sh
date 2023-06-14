prompt=$1
best_response=$2

python3 -m torch.distributed.run --nproc_per_node=6 --master_port=1234 /scratch/data/karan/alpaca_farm/examples/test_destroy/destroy_sft.py \
  --model_name_or_path "/scratch/data/karan/models/alpaca_farm_models/sft10k" \
  --fp16 False \
  --bf16 True \
  --seed 42 \
  --output_dir "/scratch/data/karan/alpaca_farm/examples/test_destroy/model10" \
  --num_train_epochs 10 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 1000000000 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --logging_strategy epoch \
  --wandb_project "alpaca_farm" \
  --run_name "SFT Destroy" \
  --tf32 False \
  --flash_attn True \
  --model_max_length 512 \
  --ddp_timeout 1800 \
  --fsdp "full_shard auto_wrap" \
  --do_eval False \
  --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
  --prompt "${prompt}" \
  --best_response "${best_response}"