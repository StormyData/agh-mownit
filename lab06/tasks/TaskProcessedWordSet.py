import json
import pathlib
import pickle
from collections import Counter

from nltk.corpus import stopwords
from curses import ascii

from tasks.TaskMasterWordSet import TaskMasterWordSet
from tasks.BaseTask import Task

sw = set(stopwords.words("english"))


def test_ascii(word: str):
    return all(ascii.isascii(char) and ascii.isalpha(char) for char in word)


def test(word: str):
    return word not in sw and test_ascii(word)


class TaskProcessedWordSet(Task):
    cfg_path = pathlib.Path("config/TaskProcessedWordSet.json")

    @staticmethod
    def get_requirements():
        return [TaskMasterWordSet]

    @classmethod
    def last_update_time(cls):
        output_modification_time = Task.get_path_modification_time("data/processed_word_set.pickle")
        config_modification_time = Task.get_path_modification_time(cls.cfg_path)
        if config_modification_time is None or output_modification_time is None or output_modification_time < config_modification_time:
            return None
        return output_modification_time

    @classmethod
    def execute(cls):
        if not cls.cfg_path.exists() or not cls.cfg_path.is_file():
            raise EnvironmentError("config file doesn't exist")
        with open(cls.cfg_path) as f:
            cfg = json.load(f)
            cfg: dict

        take = cfg['take']

        with open("data/master_word_counter.pickle", "rb") as f:
            word_counter = pickle.load(f)
            word_counter: Counter
        arr = sorted(filter(lambda x: test(x[0]), word_counter.items()), key=lambda x: x[1])
        to_remove = (len(arr) - take) // 2

        with open("data/processed_word_set.pickle", "wb") as f:
            pickle.dump(set(word for word, _ in arr[to_remove:-to_remove]), f)