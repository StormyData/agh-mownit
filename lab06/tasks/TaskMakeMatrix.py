import pathlib
import pickle
from collections import Counter

import scipy.sparse
import tqdm

from tasks.BaseTask import Task
from tasks.TaskEnumeratedWords import TaskEnumeratedWords
from tasks.TaskIndexToUUID import TaskIndexToUUID
from tasks.TaskPageWordCount import TaskPageWordCount


class TaskMakeMatrix(Task):
    @staticmethod
    def get_requirements():
        return [TaskEnumeratedWords, TaskIndexToUUID, TaskPageWordCount]

    @staticmethod
    def last_update_time():
        return Task.get_path_modification_time("data/full_matrix_before_IDF.pickle")

    @staticmethod
    def execute():
        with open("data/enumerated_words.pickle", "rb") as f:
            word_dict = pickle.load(f)
            word_dict: dict

        with open("data/index_to_uuid.pickle", "rb") as f:
            index_mapping = pickle.load(f)

        m = len(word_dict.keys())
        n = len(index_mapping.values())

        matrix = scipy.sparse.dok_matrix((m, n), dtype='float')

        for col, file in tqdm.tqdm(index_mapping.items(), desc="making matrix"):
            with open(pathlib.Path("data/words").joinpath(file), "rb") as f:
                counter = pickle.load(f)
                counter: Counter
            for key, val in counter.items():
                if key not in word_dict:
                    continue
                row = word_dict[key]
                matrix[row, col] = val

        with open("data/full_matrix_before_IDF.pickle", "wb") as f:
            pickle.dump(matrix, f)