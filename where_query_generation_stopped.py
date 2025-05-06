import json
import os
import sys
from tqdm import tqdm

htr_dir = sys.argv[1]
endtext = sys.argv[2]
dirlist = os.listdir(htr_dir)

checkpoint = False

for subdir in tqdm(dirlist):
    the_text = ""
    subdir_path = os.path.join(htr_dir, subdir)
    for filename in os.listdir(subdir_path):
        if checkpoint:
            if filename.endswith(".json"):
                filepath = os.path.join(subdir_path, filename)
                with open(filepath, "r", encoding="utf-8") as file:
                    data = json.load(file)
                try:
                    for part in data["contains"]:
                        for text in part["contains"]:
                            the_text += " ".join(text["text_result"]["texts"])+" "
                except KeyError:
                    pass
                if endtext in the_text:
                    print("Here:", filepath)
                    exit()
        if filename == "A0068689_00710.json":
            checkpoint = True