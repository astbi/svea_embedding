import os
import sys
import getpass
import random
import json
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

def get_sequence_length(sequence_lengths, length_weigths):
    """ For sampling the length of a text sequence """
    length_span = random.choices(sequence_lengths, weights=length_weigths, k=1)[0]
    length = random.randint(length_span[0], length_span[1])
    return length

def prepare_htr_data(input_dir):
    """ Read HTR files in subdirs and gather the text of all pages"""
    volumes = []
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        volume_text = ""
        for filename in os.listdir(subdir_path):
            if filename.endswith(".json"):
                filepath = os.path.join(subdir_path, filename)
                with open(filepath, "r", encoding="utf-8") as file:
                    data = json.load(file)
                try:
                    for part in data["contains"]:
                        for text in part["contains"]:
                            volume_text += " ".join(text["text_result"]["texts"])+" "
                except KeyError: # not a normal file
                    pass
        volume_text = volume_text.replace("¬ ", "") # ta bort radbrytningstecken
        volumes.append(volume_text)
    return volumes

# Reading txt files and gathering data
# TODO Hur ta bort krokar?
def read_data(input_dir, sequence_lengths, length_weigths):
    """ Read HTR-scanned json files in subdirectories and gather the data (strings)
     of varying length """
    print("Reading files...")
    text_data = []
    volumes = prepare_htr_data(input_dir) # list of volume strings
    print("Number of volumes:", len(volumes))
    print("Preparing text data from files...")
    # split texts into "tokens" (whitespace tokenization)
    split_volumes = [volume.split(" ") for volume in volumes]
    for i in range(len(volumes)):
        while len(split_volumes[i]) > 0:
            # Sample text lengths
            length = get_sequence_length(sequence_lengths, length_weigths)
            # Get the text with the sampled number of tokens
            text = " ".join(split_volumes[i][:length])
            if length > len(split_volumes[i]): # if there is not enough text left
                try: # take tokens from next volume
                    tokens_left = length-len(split_volumes[i])
                    text += " ".join(split_volumes[i+1][:tokens_left])
                    split_volumes[i+1] = split_volumes[i+1][tokens_left:]
                except IndexError: # if we're at the last volume
                    pass
            # update content to what is left
            split_volumes[i] = split_volumes[i][length:]
            # Add text to data list
            text_data.append(text)
    print(f"Number of texts: {len(text_data)}")
    return text_data

if __name__ == "__main__":
    print("Preparing unsupervised data")
    input_dir = sys.argv[1] # directory with txt-files
    # Lengths sampled based on the paper by Chen et al. (2024)
    sequence_lengths = [(10, 999),(1000,1999),(2000,2999),(3000,3999),(4000,4999), (5000,8192)]
    length_weigths = [.639,.257,.059,.02,.009,.016]
    text_data = read_data(input_dir, sequence_lengths, length_weigths)

    # Setting up the LLM
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    llm = init_chat_model("gpt-4o-mini", model_provider="openai")

    # Setting up the prompt template
    prompt_template = PromptTemplate.from_template(
        "Here is an old text in Swedish: {text} Translate it into modern Swedish. Answer with the translation only.")

    print("Generating translations...")
    #id = 0
    with open("unsupervised_data.jsonl", "w", encoding="utf-8") as outfile:        
        for text in text_data:
            prompt = prompt_template.invoke({"text": text})
            # This next step costs money per tokens !!!
            translation = llm.invoke(prompt).content
            # save the entry and write into outfile
            text_pair = {"query": translation, "pos": [text]}
            #id += 1
            json_line = json.dumps(text_pair, ensure_ascii=False)
            outfile.write(json_line + "\n")
    print("Done!")