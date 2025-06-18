# Svea Embedding
This repository contains scripts and example data for my master thesis in language technology: [Fine-Tuning LLMs in Information Retrieval for Historical Swedish Court Records Using AI-generated Question-Answer Data](https://uu.diva-portal.org/smash/record.jsf?dswid=-9608&pid=diva2%3A1971656&c=1&searchType=SIMPLE&language=en&query=astrid+berntsson+ingelstam&af=%5B%5D&aq=%5B%5B%5D%5D&aq2=%5B%5B%5D%5D&aqe=%5B%5D&noOfRows=50&sortOrder=author_sort_asc&sortOrder2=title_sort_asc&onlyFullText=false&sf=all).

To follow the methodology steps, one first needs to install the FlagEmbedding package:

`pip install -U FlagEmbedding[finetine]`

## Curation of Training Data
In the directory "small_data" is a small HTR-scanned volume of court record pages from the Svea Court of Appeal. The final training data has the form of the data in the directory "example_train_split".

**Generating queries:**
The first step for preparing training and evaluation data is to read the HTR-scanned archival text, split is into documents and generate queries: 

`python generate_queries.py small_data example_pos.jsonl`

The query generation is a time consuming process. If it for some reason stops before being done, one can find out where by running this command and putting in the path to the HTR volumes along with the last processed achival text.

`python where_query_generation_stopped.py small_data "text last processed"`

**Mine hard negtives:**
Next, we mine hard negatives for the training data with `hn_mine.py`:
```
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
```

**Add re-ranker scores:**
After mining hard negatives, we add re-ranker scores with `add_reranker_score.py`:
```
python add_reranker_score.py \
--input_file example_posneg.jsonl \
--output_file example_train.jsonl \
--reranker_name_or_path BAAI/bge-reranker-v2-m3 \
--reranker_query_max_length 256 \
--reranker_max_length 8192 \
--reranker_batch_size 1
```

**Split data by length:**
Finally, the training data is split by length:
```
python split_data_by_length.py \
--input_path example_train.jsonl \
--output_dir example_train_split \
--cache_dir .cache \
--log_name .split_log \
--length_list 0 500 1000 2000 3000 4000 5000 6000 7000 \
--model_name_or_path BAAI/bge-m3 \
--num_proc 16 \
```

The scripts `hn_mine`, `add_reranker_score` and `split_data_by_length` were copied from the Flag Embedding GitHub repository (https://github.com/FlagOpen/FlagEmbedding) and modified to handle utf-8 encoding.

## Fine-tuning of BGE M3-Embedding
We follow the fine-tuning instructions of BGE M3-Embedding availibe in the FlagEmbedding GitHub repository: https://github.com/FlagOpen/FlagEmbedding.

To fine-tune the model, we run the following command, but with the corret path to our training data.
```
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
	--output_dir ./svea_embedding \
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
```

## Gold Standard
Once we have queries and documents for evaluation, we can prepare the gold standard ranking. The input examples need to have the same for as in the file `example_pos.jsonl`: queries paired with one positive document.

`python prepare_eval_data.py example_pos.jsonl example_gold_standard.jsonl`

## Evaluation
The following command measures retrieval performance according to our experiments and writes the resulting scores in a file called `scores.txt`.
```
python evaluation.py \
--test_data_file example_pos.jsonl \
--goldfile example_gold_standard.jsonl \
--k_documents 100 \
--methods bm25 \
	BAAI/bge-m3 \
	KBLab/sentence-bert-swedish-cased \
	castorini/mdpr-tied-pft-msmarco \
	gemasphi/mcontriever \
	intfloat/multilingual-e5-small
--local_m3_model svea_embedding/...
```
`--methods` is a list of the retrieval methods that get evaluated, either `bm25` or the path to a HuggingFace model.
`--local_m3_model` is an optional argument and should contain the path to the checkpoint of a local fine-tuned M3-Embedding model.

For the experiments with different document lengths, we filter the testdata file and goldfile to include only short or long documents.
`python split_testdata.py example_pos.jsonl example_gold_standard.jsonl`
