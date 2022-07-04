import json
import math
import pickle

import tqdm
import scipy.sparse
import pathlib
from collections import Counter

from tasks.BaseTask import Task
from tasks.TaskCheckpointDict import TaskCheckpointDict
from tasks.TaskEnumeratedWords import TaskEnumeratedWords
from tasks.TaskExistIn import TaskExistsIn
from tasks.TaskIndexToUUID import TaskIndexToUUID
from tasks.TaskPageWordCount import TaskPageWordCount


class TaskMakeIDFMatrix(Task):
    @staticmethod
    def get_requirements():
        return [TaskIndexToUUID, TaskPageWordCount, TaskExistsIn, TaskEnumeratedWords]

    @staticmethod
    def last_update_time():
        return Task.get_path_modification_time("data/full_matrix.pickle")

    @staticmethod
    def execute():
        with open("data/number_of_docs_words_exist_in.pickle", "rb") as f:
            exists_in = pickle.load(f)

        with open("data/enumerated_words.pickle", "rb") as f:
            enumerated_words = pickle.load(f)

        with open("data/index_to_uuid.pickle", "rb") as f:
            index_mapping = pickle.load(f)

        m = len(enumerated_words.keys())
        n = len(index_mapping.values())

        matrix = scipy.sparse.dok_matrix((m, n), dtype='float')

        for col, file in tqdm.tqdm(index_mapping.items(), desc="calculating IDF matrix"):
            with open(pathlib.Path("data/words").joinpath(file), "rb") as f:
                counter = pickle.load(f)
                counter: Counter
            for word, count in counter.items():
                if word not in enumerated_words:
                    continue
                ne = exists_in[word]
                row = enumerated_words[word]
                if ne > 0:
                    matrix[row, col] = count * math.log(n / ne)

        with open("data/full_matrix.pickle", "wb") as f:
            pickle.dump(matrix, f)
