import pickle

from tasks.BaseTask import Task
from tasks.TaskProcessedWordSet import TaskProcessedWordSet


class TaskEnumeratedWords(Task):
    @staticmethod
    def get_requirements():
        return [TaskProcessedWordSet]

    @staticmethod
    def last_update_time():
        return Task.get_path_modification_time("data/enumerated_words.pickle")

    @staticmethod
    def execute():
        with open("data/processed_word_set.pickle", "rb") as f:
            word_set = pickle.load(f)
            word_set: set[str]
        sorted_words = sorted(word_set)
        word_dict = {word: index for index, word in enumerate(sorted_words)}
        with open("data/enumerated_words.pickle", "wb") as f:
            pickle.dump(word_dict, f)