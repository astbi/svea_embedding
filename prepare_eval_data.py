import getpass
import os
import json
import random
import sys
from langchain_core.vectorstores import InMemoryVectorStore
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings

testdata_pos_path = sys.argv[1]
output_path = sys.argv[2]

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# Load the embedder model and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

# Gather documents and queries
queries = []
print("Gathering data...")
with open(testdata_pos_path,"r",encoding="utf-8") as infile:
  for line in tqdm(infile):
    content = json.loads(line)
    query = content["query"]
    queries.append(query)
    text = content["pos"] # list with one text
    vector_store.add_texts(text) # embed text

ratio = 0.005 # 0.5% of results will be saved for manual evaluation
random_i = 1

with open(output_path,"w",encoding="utf-8") as outfile:
    print("Searching for each query...")
    for query in tqdm(queries):
        results = vector_store.similarity_search(query, k=10) # perform search for 10 best documents
        data = {"query": query, "documents": [r.page_content for r in results]} # save query and texts
        json_line = json.dumps(data, ensure_ascii=False)
        outfile.write(json_line + "\n")
        if random.random() < ratio: # write some random results in separate files as well
           with open("testdata_to_inspect/results_"+str(random_i)+".txt", "w", encoding="utf-8") as rf:
              rf.write(query+"\n"+str([r.page_content for r in results[:3]])) # 3 best documents
        random_i += 1