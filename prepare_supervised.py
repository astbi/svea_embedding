import sys
import os
import json
import getpass
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from prepare_unsupervised import read_data

print("Preparing query-pos pairs")

# Read txt files and gather positive text data
input_dir = sys.argv[1]
pos_data_lst = read_data(input_dir,
                         sequence_lengths=[(10,499), (500, 999),(1000,1999),(2000,2999),(3000,3999),(4000,4999), (5000,5999), (6000,6999), (7000,8192)],
                         length_weigths=[85.850, 8.872, 2.053, 0.323, 0.107, 0.033, 0.040, 0.152, 2.568]
                         )

# Generate a query for each text
# Setting up the LLM
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Setting up the prompt template
prompt_template = PromptTemplate.from_template(
    "Here is an old text in Swedish: {text} Generate a question in modern Swedish that can be answered by the text, as specific to the text's content as possible and without using the word 'text'. Answer with the question only.")

print("Generating queries...")
queries = []
for text in pos_data_lst:
    prompt = prompt_template.invoke({"text": text})
    # This next step costs money per tokens !!!
    query = llm.invoke(prompt).content
    queries.append(query)

assert len(pos_data_lst) == len(queries)

# Write data into jsonl-file
print("Writing to file...")
with open("supervised_pos.jsonl", "w", encoding="utf-8") as outfile:
    for i in range(len(pos_data_lst)):
        text_pair = {"query": queries[i], "pos": [pos_data_lst[i]]}
        json_line = json.dumps(text_pair, ensure_ascii=False)
        outfile.write(json_line + "\n")
print("Done!")
# TODO
# TQDM
# sätt upp träning