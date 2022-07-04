import pickle
import json
import pathlib
import shutil

import scipy.sparse
import scipy.sparse.linalg

from tasks.BaseTask import Task
from tasks.TaskMakeMatrix import TaskMakeMatrix
from tasks.TaskMakeIDFMatrix import TaskMakeIDFMatrix


class TaskSVDMatrix(Task):
    cfg_path = pathlib.Path("config/TaskSVDMatrix.json")
    @staticmethod
    def get_requirements():
        return [TaskMakeMatrix, TaskMakeIDFMatrix]

    @classmethod
    def last_update_time(cls):
        result_mt = Task.get_path_modification_time("data/SVDMatrix.pickle")
        config_mt = Task.get_path_modification_time(cls.cfg_path)
        if result_mt is None or config_mt is None or result_mt < config_mt:
            return None
        return result_mt

    @classmethod
    def execute(cls):
        if not cls.cfg_path.exists() or not cls.cfg_path.is_file():
            raise EnvironmentError("config file doesn't exist")
        with open(cls.cfg_path) as f:
            cfg = json.load(f)
            cfg: dict
        match cfg['matrix_type']:
            case "IDF":
                matrix_path = "data/full_matrix.pickle"
            case "NOIDF":
                matrix_path = "data/full_matrix_before_IDF.pickle"
            case _:
                raise ValueError("incorrect config parameter matrix_type")
        k = cfg['k']
        with open(matrix_path, "rb") as f:
            matrix = pickle.load(f)
            matrix: scipy.sparse.dok_matrix
        print("performing svd decomposition")
        u, s, vh = scipy.sparse.linalg.svds(matrix, k)
        print("finished svd decomposition")
        with open("data/SVDMatrix.pickle", "wb") as f:
            pickle.dump((u, s, vh), f)