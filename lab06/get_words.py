#!/bin/python3
import pickle

with open("data/processed_word_set.pickle","rb") as f:
    ws = pickle.load(f)
with open("words.txt", "w") as f:
    f.write("\n".join(ws))
