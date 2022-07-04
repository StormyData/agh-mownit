import pickle

import numpy as np
import tqdm

import scipy.sparse
import scipy.sparse.linalg

from tasks.BaseTask import Task
from tasks.TaskMakeIDFMatrix import TaskMakeIDFMatrix


class TaskNormalizedMatrix(Task):
    @staticmethod
    def get_requirements():
        return [TaskMakeIDFMatrix]

    @staticmethod
    def last_update_time():
        return Task.get_path_modification_time("data/NormalizedMatrix.pickle")

    @staticmethod
    def execute():
        with open("data/full_matrix.pickle", "rb") as f:
            matrix = pickle.load(f)
            matrix: scipy.sparse.dok_matrix
        m, n = matrix.shape
        out = np.empty(n)
        for i in tqdm.trange(n):
            vec = np.zeros(n)
            vec[i] = 1
            norm = np.linalg.norm(matrix @ vec)
            if np.isclose(norm, 0):
                out[i] = 0
            else:
                out[i] = 1.0 / norm
        matrix = matrix @ scipy.sparse.diags(out)
        with open("data/NormalizedMatrix.pickle", "wb") as f:
            pickle.dump(matrix, f)
