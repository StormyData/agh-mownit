import json
import pathlib
import pickle
from collections import Counter
from multiprocessing import Pool

import tqdm
from bs4 import BeautifulSoup
from nltk import word_tokenize

from tasks.BaseTask import Task
from tasks.TaskCheckpointDict import TaskCheckpointDict
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def extract_words(file):
    src_path = pathlib.Path("data/pages").joinpath(file)
    src_mtime = Task.get_path_modification_time(src_path)
    dst_path = pathlib.Path("data/words").joinpath(file)
    dst_mtime = Task.get_path_modification_time(dst_path)
    if src_path is None or (dst_mtime is not None and src_mtime < dst_mtime):
        return
    word_counter = Counter()
    with open(src_path) as f:
        soup = BeautifulSoup(f, 'html.parser')
    for words in soup.stripped_strings:
        word_counter += Counter(ps.stem(word, to_lowercase=True) for word in word_tokenize(words))
    with open(dst_path, "wb") as f2:
        pickle.dump(word_counter, f2)


class TaskPageWordCount(Task):
    @staticmethod
    def get_requirements():
        return [TaskCheckpointDict]

    @staticmethod
    def last_update_time():
        return Task.get_path_modification_time("data/TaskPageWordCount.dummy")

    @staticmethod
    def execute():
        with open("data/checkpoint_dict.json") as f:
            mapping = json.load(f)
            mapping: dict
        chunksize = 100
        with Pool() as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(extract_words, mapping.values(), chunksize=chunksize),
                               total=len(mapping.values())):
                pass
        pathlib.Path("data/TaskPageWordCount.dummy").touch()