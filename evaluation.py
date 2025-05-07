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

class Evaluation():
    def __init__(self, datafile, goldfile, k, include_map):
        self.queries, self.documents = self._gather_data(datafile)
        self.k = k # number of documents to retrieve for each query
        self.include_map = include_map
        self.gold_standard = self._get_gold_standard(goldfile)
        self.n_relevant = len(list(self.gold_standard.values())[0]) # number of relevant documents for each query
    
    def _gather_data(self, path):
        """Gathers all queries and documents in the test data"""
        documents = []
        queries = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                queries.append(data["query"])
                for doc in data["pos"]:
                    documents.append(doc)
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
                query = data["query"]
                relevant_documents = data["documents"]
                n = len(relevant_documents)
                res = [(relevant_documents[i], n-i) for i in range(n)]
                results[query] = res
        return results
    
    def _get_mean(self, L):
        """Helper function for scoring that returns the mean value
        of a list of floats"""
        return sum(L)/len(L)
    
    def retrieve_hf_model(self, model_path):
        """ Retrieving with a model from huggingface. Returns result dict
        {query: [(doc1, score1), doc2,score2]...} and MAP"""
        print(f"Retrieval with {model_path}...")
        torch.cuda.empty_cache()
        model = SentenceTransformer(model_path) # load model
        print("Embedding documents...")
        document_embeddings = model.encode(self.documents, convert_to_numpy=True) # encode corpus
        dimension = document_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension) # create faiss index
        index.add(document_embeddings)
        results = dict()
        if self.include_map:
            average_precisions = []
        print("Retrieving documents for each query...")
        for query in tqdm(self.queries): # get relevant documents and scores for each query
            query_embedding = model.encode([query], convert_to_numpy=True) # embed query
            distances, indices = index.search(query_embedding, k=len(self.documents)) # search
            # Top k results with document and score 
            top_k_results = [(self.documents[indices[0][i]], float(distances[0][i])) for i in range(self.k)]
            results[query] = top_k_results # save top k
            # Calculate average precision with sorted document list
            if self.include_map:
                sorted_docs = [self.documents[indices[0][i]] for i in range(len(self.documents))]
                average_precisions.append(self._average_precision(sorted_docs, query))
        if self.include_map:
            map = self._get_mean(average_precisions)
        else:
            map=None
        return results, map

    def _average_precision(self, sorted_results, query):
        """Calculate average precision for one information need (query).
        sorted_results = a list of documents in order of relevance 
        judged by a model or BM25."""
        gold_docs = [res[0] for res in self.gold_standard[query]] # list of relevant documents
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
        if self.include_map:
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
            # sorted document list for calculating average precision
            if self.include_map:
                sorted_docs = [self.documents[i] for i in sorted_indices]
                average_precisions.append(self._average_precision(sorted_docs, query))      
        if self.include_map:
            map = self._get_mean(average_precisions)
        else:
            map=None
        return results, map

    def precision_recall(self, results):
        """ Calculate precision and recall  @ k from results dict"""
        recalls = []
        precisions = []
        for query in self.queries:
            n_relevant_retrieved = 0
            gold_docs = [res[0] for res in self.gold_standard[query]] # true documents
            for res in results[query]:
                if res[0] in gold_docs:
                    n_relevant_retrieved += 1
            recalls.append(n_relevant_retrieved/self.n_relevant)
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
            doc_truerelevance = self._doc_relevance_dict(self.gold_standard[query])
            doc_relevancescore = self._doc_relevance_dict(results[query])
            n_documents = len(self.documents)
            # Create numpy arrays to be compared
            true_relevance = np.zeros((1, n_documents))
            relevance_scores = np.zeros((1, n_documents))
            # Add relevance scores from dictionaries to arrays
            for i in range(n_documents):
                document = self.documents[i]
                true_relevance[0][i] = doc_truerelevance[document]
                relevance_scores[0][i] = doc_relevancescore[document]
            # Compare
            ndcgs.append(ndcg_score(true_relevance, relevance_scores))
        return self._get_mean(ndcgs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_file", type=str, required=True)
    parser.add_argument("--goldfile", type=str, required=True)
    parser.add_argument("--outfile", type=str, default="scores.txt")
    parser.add_argument("--k_documents", type=int, default=100, help="Number of documents to retrieve for each query.")
    parser.add_argument("--include_map", type=bool, default=True)
    parser.add_argument("--hf_models", nargs="+", default=["BAAI/bge-m3", 
                                                           "KBLab/sentence-bert-swedish-cased", 
                                                           "castorini/mdpr-tied-pft-msmarco", 
                                                           "gemasphi/mcontriever", 
                                                           "intfloat/multilingual-e5-small"])
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
                            k=args.k_documents,
                            include_map = args.include_map)
    
    with open(args.outfile, "w", encoding="utf-8") as outfile:
        bm25_results, bm25_map = evaluation.retrieve_bm25()
        outfile.write(f"BM25\n")
        if args.include_map:
            outfile.write(f"MAP: {bm25_map*100}\n")
        for s in (score_and_print(evaluation, bm25_results)):
            outfile.write(s)
        
        for model in args.hf_models:
            outfile.write(f"{model}\n")
            try:
                model_results, model_map = evaluation.retrieve_hf_model(model)
                if args.include_map:
                    outfile.write(f"MAP: {model_map*100}%\n")
                for s in score_and_print(evaluation, model_results):
                    outfile.write(s)
            except torch.cuda.OutOfMemoryError:
                outfile.write("Not enough memory.\n\n")