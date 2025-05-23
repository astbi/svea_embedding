import json
import os
import sys
from tqdm import tqdm

htr_dir = sys.argv[1] # directory with HTR volumes
endtext = sys.argv[2] # last text string processed
dirlist = os.listdir(htr_dir)

for subdir in tqdm(dirlist):
    the_text = "" # For gathering text
    subdir_path = os.path.join(htr_dir, subdir)
    for filename in os.listdir(subdir_path):
        if filename.endswith(".json"):
            filepath = os.path.join(subdir_path, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)
            try:
                for part in data["contains"]:
                    for text in part["contains"]:
                        # Add the text from the file
                        the_text += " ".join(text["text_result"]["texts"])+" "
            except KeyError:
                pass
            if endtext in the_text: # We found the text
                print("Here:", filepath)
                exit()
