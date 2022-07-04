import json
import pathlib
import pickle
from collections import Counter

import tqdm

from tasks.BaseTask import Task
from tasks.TaskCheckpointDict import TaskCheckpointDict
from tasks.TaskMasterWordSet import TaskMasterWordSet
from tasks.TaskPageWordCount import TaskPageWordCount


class TaskExistsIn(Task):
    @staticmethod
    def get_requirements():
        return [TaskMasterWordSet, TaskCheckpointDict, TaskPageWordCount]

    @staticmethod
    def last_update_time():
        return Task.get_path_modification_time("data/number_of_docs_words_exist_in.pickle")

    @staticmethod
    def execute():
        with open("data/master_word_set.pickle", "rb") as f:
            word_set = pickle.load(f)
            word_set: set[str]

        with open("data/checkpoint_dict.json", "rb") as f:
            uuids = json.load(f).values()

        exists_in = {word: 0 for word in word_set}
        for file in tqdm.tqdm(uuids, desc="building 'exists in' database"):
            with open(pathlib.Path("data/words").joinpath(file), "rb") as f:
                counter = pickle.load(f)
                counter: Counter
            for key in counter.keys():
                exists_in[key] += 1
        with open("data/number_of_docs_words_exist_in.pickle", "wb") as f:
            pickle.dump(exists_in, f)