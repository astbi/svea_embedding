#!/bin/bash
eval "$(/c/Users/Admin/miniconda3/Scripts/conda.exe shell.bash hook)"
conda activate thesis_prepro

python prepare_unsupervised.py small_data

echo "Splitting unsupervised data..."

python split_unsupervised.py unsupervised_data.jsonl

python prepare_supervised.py small_data

git clone https://github.com/FlagOpen/FlagEmbedding
cd FlagEmbedding/scripts

echo "Mining negatives..."
python hn_mine.py \
--input_file c:/Users/Admin/Thesis/supervised_pos.jsonl \
--output_file c:/Users/Admin/Thesis/supervised_posneg.jsonl \
--range_for_sampling 2-200 \
--negative_number 1 \
--embedder_name_or_path BAAI/bge-m3

echo "Adding teacher scores..."
python add_reranker_score.py \
--input_file c:/Users/Admin/Thesis/supervised_posneg.jsonl \
--output_file c:/Users/Admin/Thesis/supervised_data.jsonl \
--reranker_name_or_path BAAI/bge-reranker-v2-m3 \
--reranker_query_max_length 512 \
--reranker_max_length 8192

echo "Splitting supervised data by length..."
python split_data_by_length.py \
--input_path c:/Users/Admin/Thesis/supervised_data.jsonl \
--output_dir c:/Users/Admin/Thesis/supervised_data_split \
--cache_dir .cache \
--log_name .split_log \
--length_list 0 500 1000 2000 3000 4000 5000 6000 7000 \
--model_name_or_path BAAI/bge-m3 \
--num_proc 16 \

echo "Done!!!"