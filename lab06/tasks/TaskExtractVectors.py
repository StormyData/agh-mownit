import pickle
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import tqdm
from multiprocessing import Pool

from tasks.BaseTask import Task
from tasks.TaskSVDMatrix import TaskSVDMatrix


def get_weight(arg):
    u, s, vh, n, i = arg
    vec = np.zeros(n)
    vec[i] = 1
    vec = u @ (scipy.sparse.diags(s) @ (vh @ vec))
    norm = np.linalg.norm(vec)
    if np.isclose(norm, 0):
        return i, 0
    return i, 1.0 / norm


class TaskExtractVectors(Task):
    @staticmethod
    def get_requirements():
        return [TaskSVDMatrix]

    @staticmethod
    def last_update_time():
        return Task.get_path_modification_time("data/SVD_with_vectors.pickle")

    @staticmethod
    def execute():
        with open("data/SVDMatrix.pickle", "rb") as f:
            u, s, vh = pickle.load(f)
        n = vh.shape[1]
        weights_arr = np.empty(n)
        with Pool() as pool:
            for ret in tqdm.tqdm(pool.imap_unordered(get_weight, ((u, s, vh, n, i) for i in range(n)), chunksize=100),
                                 total=n):
                weights_arr[ret[0]] = ret[1]

        with open("data/SVD_with_vectors.pickle", "wb") as f:
            pickle.dump((u, s, vh, weights_arr), f)
