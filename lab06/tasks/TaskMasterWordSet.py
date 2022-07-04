import json
import pathlib
import pickle
from collections import Counter

import tqdm

from tasks.BaseTask import Task
from tasks.TaskPageWordCount import TaskPageWordCount


class TaskMasterWordSet(Task):
    @staticmethod
    def get_requirements():
        return [TaskPageWordCount]

    @staticmethod
    def last_update_time():
        return Task.get_path_modification_time("data/master_word_set.pickle")

    @staticmethod
    def execute():
        with open("data/checkpoint_dict.json") as f:
            mapping = json.load(f)
            mapping: dict
        master_counter = Counter()
        for file in tqdm.tqdm(mapping.values()):
            with open(pathlib.Path("data/words").joinpath(file), "rb") as f2:
                master_counter += pickle.load(f2)

        with open("data/master_word_counter.pickle", "wb") as f:
            pickle.dump(master_counter, f)

        with open("data/master_word_set.pickle", "wb") as f:
            pickle.dump(set(master_counter.keys()), f)