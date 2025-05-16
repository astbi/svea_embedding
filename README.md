# ra_m3_masterthesis
This repository contains scripts and example data for my masterthesis.

## Pre-processing of Training Data
I have includes some example data in the directory "small_data". The final training data has the form of the data in the directory "example_train_split".

Generate queries:
python generate_queries.py small_data example_pos.jsonl

Mine hard negtives:
python hn_mine.py \
--input_file example_pos.jsonl \
--output_file example_posneg.jsonl \
--range_for_sampling 2-200 \
--negative_number 7 \
--embedder_name_or_path BAAI/bge-m3 \
--use_gpu_for_searching True \
--search_batch_size 1 \
--batch_size 1 \
--embedder_passage_max_length 8192 \
--embedder_query_max_length 256

Add teacher scores:
python add_reranker_score.py \
--input_file example_posneg.jsonl \
--output_file example_train.jsonl \
--reranker_name_or_path BAAI/bge-reranker-v2-m3 \
--reranker_query_max_length 256 \
--reranker_max_length 8192 \
--reranker_batch_size 1

Split data by length:
python split_data_by_length.py \
--input_path example_train.jsonl \
--output_dir example_train_split \
--cache_dir .cache \
--log_name .split_log \
--length_list 0 500 1000 2000 3000 4000 5000 6000 7000 \
--model_name_or_path BAAI/bge-m3 \
--num_proc 16 \

## Fine-tuning of BGE M3-Embedding
torchrun --nproc_per_node 2 \
	-m FlagEmbedding.finetune.embedder.encoder_only.m3 \
	--model_name_or_path BAAI/bge-m3 \
    --cache_dir ./cache/model \
    --train_data ./example_supervised_data_split \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 256 \
    --passage_max_len 8192 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation True \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
    --output_dir ./bge_m3_finetuned \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
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
    --self_distill_start_step 0 

## Evaluation
