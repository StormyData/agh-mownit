import pathlib


class Task:
    @staticmethod
    def get_path_modification_time(path: str | pathlib.Path):
        path = pathlib.Path(path)
        if not path.exists():
            return None
        return path.stat().st_mtime

    @staticmethod
    def get_requirements():
        return []

    @staticmethod
    def last_update_time():
        return None

    @staticmethod
    def execute():
        pass