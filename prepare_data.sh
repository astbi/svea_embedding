#!/bin/bash
eval "$(/c/Users/Admin/miniconda3/Scripts/conda.exe shell.bash hook)"
conda activate thesis_prepro

python generate_queries.py small_data example_pos.jsonl

echo "Mining negatives..."
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

echo "Adding teacher scores..."
python add_reranker_score.py \
--input_file example_posneg.jsonl \
--output_file example_train.jsonl \
--reranker_name_or_path BAAI/bge-reranker-v2-m3 \
--reranker_query_max_length 256 \
--reranker_max_length 8192 \
--reranker_batch_size 1

echo "Splitting supervised data by length..."
python split_data_by_length.py \
--input_path example_train.jsonl \
--output_dir example_train_split \
--cache_dir .cache \
--log_name .split_log \
--length_list 0 500 1000 2000 3000 4000 5000 6000 7000 \
--model_name_or_path BAAI/bge-m3 \
--num_proc 16 \

echo "Done!"