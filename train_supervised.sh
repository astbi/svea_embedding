#!/bin/bash -l
#SBATCH -A uppmax2020-2-2
#SBATCH -p node
#SBATCH -M snowy # cluster name
#SBATCH -N 1 # resource
#SBATCH -t 00:05:00 # time
#SBATCH -J supervised # job name
#SBATCH --gpus=1
#SBATCH -q gpu
#SBATCH -o supervised.out
#SBATCH -e supervised.err

source ~/.bashrc
conda activate thesis_train

torchrun --nproc_per_node 1 \
	-m FlagEmbedding.finetune.embedder.encoder_only.m3 \
	--model_name_or_path BAAI/bge-m3 \
    --cache_dir ./cache/model \
    --train_data ./example_supervised_data_split \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 8192 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation True \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
    --output_dir ./ra_m3 \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ds_stage0.json \
    --logging_steps 1 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type m3_kd_loss \
    --unified_finetuning True \
    --use_self_distill True \
    --fix_encoder False \
    --self_distill_start_step 0 \
    --torch_empty_cache_steps 1 \
    --split_batches True