from tasks.BaseTask import Task


class TaskCheckpointDict(Task):
    @staticmethod
    def last_update_time():
        return Task.get_path_modification_time("data/checkpoint_dict.json")