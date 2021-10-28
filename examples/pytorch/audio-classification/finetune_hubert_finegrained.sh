python run_audio_classification.py \
	--model_name_or_path superb/hubert-base-superb-ks \
	--dataset_name superb \
	--dataset_config_name ks \
	--output_dir hubert_ks_finetune_finegrained \
	--overwrite_output_dir \
	--remove_unused_columns False \
	--do_train --eval_split_name test --learning_rate 1e-3 \
	--max_length_seconds 1 \
	--warmup_ratio 0.1 \
	--num_train_epochs 200 \
	--per_device_train_batch_size 32 \
	--gradient_accumulation_steps 4 \
	--per_device_eval_batch_size 32 \
	--dataloader_num_workers 12 \
	--logging_strategy steps \
	--logging_steps 10 \
	--evaluation_strategy epoch \
	--save_strategy epoch \
	--load_best_model_at_end True \
	--save_total_limit 3 \
	--seed 0 \
	--finegrained \
	--sparsity_ratio 0.95
