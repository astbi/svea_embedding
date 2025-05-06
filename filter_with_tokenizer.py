from transformers import AutoTokenizer
import json
import sys
from tqdm import tqdm

in_path = sys.argv[1]
out_path = sys.argv[2]

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
too_long_count = 0

underfivehundred = 0
underonek = 0
undertwok = 0
underthreek = 0
underfourk = 0
underfivek = 0
undersixk = 0
undersevenk = 0
oversevenk = 0

with open(in_path,"r",encoding="utf-8") as infile:
    with open(out_path, "w", encoding="utf-8") as outfile:
        for line in tqdm(infile):
            text = json.loads(line)["pos"][0] 
            tokenized = tokenizer(text)["input_ids"]
            length = len(tokenized)
            if length > 8192:
                too_long_count += 1
            else:
                outfile.write(line)
                if length < 500:
                    underfivehundred += 1
                elif length < 1000:
                    underonek += 1
                elif length < 2000:
                    undertwok += 1
                elif length < 3000:
                    underthreek += 1
                elif length < 4000:
                    underfourk += 1
                elif length < 5000:
                    underfivek += 1
                elif length < 6000:
                    undersixk += 1
                elif length < 7000:
                    undersevenk += 1
                else:
                    oversevenk += 1

print(underfivehundred, underonek, undertwok, underthreek, underfourk, underfivek, undersixk, undersevenk, oversevenk)

print("Too long texts:", too_long_count)