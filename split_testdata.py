import json
import sys
from tqdm import tqdm
from transformers import AutoTokenizer

testfile = sys.argv[1]
goldfile = sys.argv[2]
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

def write_file(path, data):
    """ Write filtered data into a new file """
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False)+"\n")

def filter_file(testfile, golddata, lengths):
    """ Filter testdatafile and goldfile to only have texts of a certain length span """
    docs_per_query = [] # for counting mean number of documents per query
    out_test_data = []
    out_gold_data = []
    with open(testfile, "r", encoding ="utf-8") as infile:
        for line in tqdm(infile):
            data = json.loads(line)
            query = data["query"]
            gold_docs = []
            # Gather gold standard documents for the query
            for doc in golddata[query]:
                tokenized_doc = tokenizer(doc)["input_ids"]
                if lengths[0] < len(tokenized_doc) < lengths[1]:
                    gold_docs.append(doc) # only keep correct length
            if gold_docs != []: # do not handle queries where all gold standard docs were filtered away
                docs_per_query.append(len(gold_docs))
                # document in testdata file
                tokenized_pos = tokenizer(data["pos"][0])["input_ids"]
                if lengths[0] < len(tokenized_pos) < lengths[1]:
                    out_test_data.append(data) # append query and document
                else: # if document gets filtered, only append query
                    out_test_data.append({"query":query,"pos":[]})
                out_gold_data.append({"query":query, "documents":gold_docs})
    write_file(f"testdata_{lengths[0]}-{lengths[1]}.jsonl", out_test_data)
    write_file(f"goldfile_{lengths[0]}-{lengths[1]}.jsonl", out_gold_data)
    return sum(docs_per_query)/len(docs_per_query) # mean number of documents per query

# Gather gold standard data in dictionary
golddata = dict()
with open(goldfile, "r", encoding="utf-8") as goldfile:
    for line in goldfile:
        data = json.loads(line)
        golddata[data["query"]] = data["documents"]

for lengths in ((0,500),(500,8192)):
    print(f"Creating split {lengths[0]}-{lengths[1]}...")
    mean_docs_per_query = filter_file(testfile, golddata, lengths)
    print(f"Mean number of documentss per query: {mean_docs_per_query}")
