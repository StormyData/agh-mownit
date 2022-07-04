import json
import pickle

from tasks.BaseTask import Task
from tasks.TaskCheckpointDict import TaskCheckpointDict


class TaskIndexToUUID(Task):
    @staticmethod
    def get_requirements():
        return [TaskCheckpointDict]

    @staticmethod
    def last_update_time():
        return Task.get_path_modification_time("data/index_to_uuid.pickle")

    @staticmethod
    def execute():
        with open("data/checkpoint_dict.json") as f:
            url_mapping = json.load(f)
            url_mapping: dict
        index_mapping = dict(enumerate(url_mapping.values()))
        with open("data/index_to_uuid.pickle", "wb") as f:
            pickle.dump(index_mapping, f)