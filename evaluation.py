import json
import faiss
import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from sklearn.metrics import ndcg_score
from FlagEmbedding import BGEM3FlagModel

class Evaluation():
    def __init__(self, datafile, goldfile, k):
        self.queries, self.documents = self._gather_data(datafile)
        self.k = k # number of documents to retrieve for each query
        self.gold_standard = self._get_gold_standard(goldfile)
    
    def _gather_data(self, path):
        """Gathers all queries and documents in the test data"""
        documents = []
        queries = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                for doc in data["pos"]:
                    documents.append(doc)
                if len(data["pos"]) > 0:
                    queries.append(data["query"])
        return queries, documents
    
    def _get_gold_standard(self, path):
        """Reads the gold standard file with relevant documents
        for each query. Returns dictionary with
        query: [(document, score), (document, score)...]. The scores
        are the descending order, so if there are 10 relevant documents
        the scores are 10,9,8,7,6,5,4,3,2,1."""
        results = dict()
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                relevant_documents = data["documents"]
                n = len(relevant_documents)
                if n > 0: # to ensure that queries without gold standard relevant documents are not tested
                    query = data["query"]
                    res = [(relevant_documents[i], n-i) for i in range(n)]
                    results[query] = res
        return results
    
    def _get_mean(self, L):
        """Helper function for scoring that returns the mean value
        of a list of floats"""
        return sum(L)/len(L)
    
    def retrieve_hf_model(self, model_path):
        """ Retrieval with a model from huggingface. Returns result dict
        {query: [(doc1, score1), doc2,score2]...} and MAP"""
        print(f"Retrieval with {model_path}...")
        torch.cuda.empty_cache()
        model = SentenceTransformer(model_path, device="cuda") # load model
        print("Embedding documents...")
        document_embeddings = model.encode(self.documents, convert_to_numpy=True, batch_size=2, show_progress_bar=True) # encode corpus
        dimension = document_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension) # create faiss index
        index.add(document_embeddings)
        results = dict()
        average_precisions = []
        print("Retrieving documents for each query...")
        for query in tqdm(self.queries): # get relevant documents and scores for each query
            query_embedding = model.encode([query], convert_to_numpy=True) # embed query
            distances, indices = index.search(query_embedding, k=len(self.documents)) # search
            # Top k results with document and score 
            top_k_results = [(self.documents[indices[0][i]], float(distances[0][i])) for i in range(self.k)]
            results[query] = top_k_results # save top k
            # Calculate average precision with sorted document list
            sorted_docs = [self.documents[indices[0][i]] for i in range(len(self.documents))]
            average_precisions.append(self._average_precision(sorted_docs, query))
        map = self._get_mean(average_precisions)
        return results, map

    def retrieve_local(self, model_path):
        """ Retrieval with a local fine-tuned BGE M3-Embedding moodel. 
        Returns result dict {query: [(doc1, score1), doc2,score2]...} and MAP"""
        print(f"Retrieval with {model_path}...")
        torch.cuda.empty_cache()
        model = BGEM3FlagModel(model_path, use_fp16=True, device="cuda") # load BGE-M3 model
        print("Embedding documents...")
        document_embeddings = [model.encode(
            doc, return_dense=True, return_sparse=True, return_colbert_vecs=True) for doc in tqdm(self.documents)] # List of document embeddings
        print("Retrieving documents for each query...")
        results = dict()
        average_precisions = []
        for query in tqdm(self.queries):
            docindices_scores = dict() # for storing the score of each document
            query_embedding = model.encode(
                query,max_length=256,return_dense=True,return_sparse=True,return_colbert_vecs=True) # encode query
            for i, doc_embedding in enumerate(document_embeddings): # go through each document embedding
                s_dense = (query_embedding["dense_vecs"] @ doc_embedding["dense_vecs"].T) # dense score
                s_sparse = model.compute_lexical_matching_score(query_embedding["lexical_weights"], doc_embedding["lexical_weights"]) # sparse score
                s_colbert = model.colbert_score(query_embedding["colbert_vecs"], doc_embedding["colbert_vecs"]) # multi-vector score
                final_score = (1 / 3 * s_dense + 1 / 3 * s_sparse + 1 / 3 * s_colbert) # combined relevance score
                docindices_scores[i] = float(final_score)
            sorted_indices = sorted(docindices_scores, key=docindices_scores.get, reverse=True) # document indices sorted by score
            top_k_results = [(self.documents[i], float(docindices_scores[i])) for i in sorted_indices[: self.k]]
            results[query] = top_k_results # save the top k documents and scores
            # Calculate average precision with sorted document list
            sorted_docs = [self.documents[i] for i in sorted_indices]
            average_precisions.append(self._average_precision(sorted_docs, query))
        map = self._get_mean(average_precisions)
        return results, map

    def _average_precision(self, sorted_results, query):
        """Calculate average precision for one information need (query).
        sorted_results = a list of documents in order of relevance 
        judged by a model or BM25."""
        gold_docs = [res[0] for res in self.gold_standard[query]] # list of true relevant documents from gold standard
        precisions = []
        for i in range(len(sorted_results)): # loop over all retrieved docs in order
            n_relevant_retrieved = 0
            for result in [sorted_results[j] for j in range(i+1)]: # retrieved docs up to index i
                if result in gold_docs:
                    n_relevant_retrieved += 1
            precisions.append(n_relevant_retrieved/(i+1)) # n relevant retrieved / n retrieved so far
            if n_relevant_retrieved == len(gold_docs):
                break # we found all relevant documents, break for-loop
        average_precision = self._get_mean(precisions)
        return average_precision

    def retrieve_bm25(self):
        """ Retrieval with BM25 using bge-m3 tokenizer. Returns result dict
        {query: [(doc1, score1), doc2,score2]...} and MAP"""
        print("Retrieval with BM25")
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3") # load tokenizer
        tokenized_corpus = [tokenizer.tokenize(doc) for doc in self.documents] # tokenize corpus
        bm25 = BM25Okapi(tokenized_corpus) # embed corpus
        results = dict()
        average_precisions = []
        print("Retrieving documents for each query...")
        for query in tqdm(self.queries):
            tokenized_query = tokenizer.tokenize(query) # tokenize query
            scores = bm25.get_scores(tokenized_query) # search
            # sort document indices after score
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            # Top k documents and scores
            top_k_results = [(self.documents[i], float(scores[i])) for i in sorted_indices[:self.k]]
            results[query] = top_k_results
            # Calculate average precision with sorted document list
            sorted_docs = [self.documents[i] for i in sorted_indices]
            average_precisions.append(self._average_precision(sorted_docs, query)) 
        map = self._get_mean(average_precisions)
        return results, map

    def precision_recall(self, results):
        """ Calculate precision@k and recall@k from results dict"""
        recalls = []
        precisions = []
        for query in self.queries:
            n_relevant_retrieved = 0
            gold_docs = [res[0] for res in self.gold_standard[query]] # true relevant documents
            n_relevant = len(gold_docs)
            for res in results[query]:
                if res[0] in gold_docs:
                    n_relevant_retrieved += 1
            recalls.append(n_relevant_retrieved/n_relevant)
            precisions.append(n_relevant_retrieved/self.k)
        return self._get_mean(precisions), self._get_mean(recalls)

    def _doc_relevance_dict(self, docs_scores_list):
        """ Helper function for ndcg() that creates a dictionary 
        {document:score} given the results of a query"""
        doc_relevance = defaultdict(float)
        for doc_score in docs_scores_list:
            doc_relevance[doc_score[0]] = float(doc_score[1])
        return doc_relevance

    def ndcg(self, results):
        """ Calculate nDCG@k from results dict"""
        ndcgs = []
        for query in self.queries:
            # Create {doc:score} dictionaries: true relevance and judged relevance scores
            doc_truerelevance = self._doc_relevance_dict(self.gold_standard[query]) # true relevance
            doc_relevancescore = self._doc_relevance_dict(results[query]) # as judged by model/bm25
            n_documents = len(self.documents)
            # Create zeros numpy arrays to be compared later
            true_relevance = np.zeros((1, n_documents))
            relevance_scores = np.zeros((1, n_documents))
            # Add relevance scores from dictionaries to the arrays
            for i in range(n_documents):
                document = self.documents[i]
                true_relevance[0][i] = doc_truerelevance[document]
                relevance_scores[0][i] = doc_relevancescore[document]
            # Compare score arrays
            ndcgs.append(ndcg_score(true_relevance, relevance_scores))
        return self._get_mean(ndcgs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_file", type=str, required=True, help="Path to testdata file {query: query, pos: [document]}")
    parser.add_argument("--goldfile", type=str, required=True, help="Path to goldfile {query: query, documents: [ranked documents]}")
    parser.add_argument("--outfile", type=str, default="scores.txt")
    parser.add_argument("--k_documents", type=int, default=100, help="Number of documents to retrieve for each query.")
    parser.add_argument("--methods", nargs="+", default=["bm25",
                                                          "BAAI/bge-m3", 
                                                           "KBLab/sentence-bert-swedish-cased", 
                                                           "castorini/mdpr-tied-pft-msmarco", 
                                                           "gemasphi/mcontriever", 
                                                           "intfloat/multilingual-e5-small"],
                                                           help="Retrieval methods that we want to evaluate: bm25 or the path to a huggingface model"
                                                           )
    parser.add_argument("--local_m3_model", type=str, required=False, help="Path to a local fine-tuned BGE M3-Embedding model")
    args = parser.parse_args()

    def score_and_print(evaluation, results):
        prec, rec = evaluation.precision_recall(results)
        ndcg = evaluation.ndcg(results)
        return [f"Recall@{evaluation.k}: {rec*100}%\n",
                f"Precision@{evaluation.k}: {prec*100}%\n",
                f"nDCG@{evaluation.k}: {ndcg*100}%\n\n",
        ]
    
    evaluation = Evaluation(datafile=args.test_data_file, 
                            goldfile=args.goldfile, 
                            k=args.k_documents)
    
    with open(args.outfile, "w", encoding="utf-8") as outfile:
        for method in args.methods:
            outfile.write(f"{method}\n")
            try:    
                if method.lower() == "bm25":
                    results, map = evaluation.retrieve_bm25()
                else:
                    results, map = evaluation.retrieve_hf_model(method)
                outfile.write(f"MAP: {map*100}%\n")
                for s in score_and_print(evaluation, results):
                    outfile.write(s)
            except torch.cuda.OutOfMemoryError as e:
                outfile.write("Not enough memory.\n\n")
                print(e)
        
        if args.local_m3_model:
            outfile.write("Local model\n")
            local_results, local_map = evaluation.retrieve_local(args.local_m3_model)
            outfile.write(f"MAP: {local_map * 100}%\n")
            for s in score_and_print(evaluation, local_results):
                outfile.write(s)