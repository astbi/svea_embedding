# ra_m3_masterthesis
För masterarbete med Riksarkivet.

small_data är några exempel från en HTR-scannad volym. 

## Preprocessing
### Unsupervised data
python prepare_unsupervised.py small_data

Dela upp efter längd:
python split_unsupervised.py unsupervised_data.jsonl

### Supervised data
python prepare_supervised.py small_data

Hard negatives:
--negative_number bör ändras till 7 när man använder större mängd data.
python hn_mine.py \
--input_file supervised_pos.jsonl \
--output_file supervised_posneg.jsonl \
--range_for_sampling 2-200 \
--negative_number 1 \
--embedder_name_or_path BAAI/bge-m3

Teacher scores:
python add_reranker_score.py \
--input_file supervised_posneg.jsonl \
--output_file supervised_data.jsonl \
--reranker_name_or_path BAAI/bge-reranker-v2-m3 \
--reranker_query_max_length 512 \
--reranker_max_length 8192

Dela upp datan efter längd:
python split_data_by_length.py \
--input_path c:/Users/Admin/Thesis/supervised_data.jsonl \
--output_dir c:/Users/Admin/Thesis/supervised_data_split \
--cache_dir .cache \
--log_name .split_log \
--length_list 0 500 1000 2000 3000 4000 5000 6000 7000 \
--model_name_or_path BAAI/bge-m3 \
--num_proc 16 \

## Träning
example_data_supervised och example_data_unsupervised innehåller liten exepeldata för träning.
train_(un)supervised.sh - för träning på Uppmax

Unsupervised träning fungerar inte i nuläget eftersom datan inte har några "neg"-exempel. Hittar ingen kod anpassad för unsupervised träning.

Supervised bör fungera men pytorch får slut på minne i Uppmax: torch.OutOfMemoryError: 
CUDA out of memory. Tried to allocate 4.65 GiB. GPU 0 has a total capacity of 14.57 GiB of which 1.03 GiB is free. Including non-PyTorch memory, this process has 13.54 GiB memory in use. Of the allocated memory 12.62 GiB is allocated by PyTorch, and 487.80 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
