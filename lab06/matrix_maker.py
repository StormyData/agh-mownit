from collections import deque
from typing import Type

from tasks.BaseTask import Task
from tasks.TaskNormalizeMatrix import TaskNormalizedMatrix
from tasks.TaskExtractVectors import TaskExtractVectors
from tasks.TaskNormalizeMatrixNoIDF import TaskNormalizedMatrixNoIDF


def linearize(start: Type[Task]) -> [Type[Task]]:
    order = {start: 0}
    to_process = deque()
    to_process.append(start)
    while len(to_process) > 0:
        curr = to_process.popleft()
        curr: Type[Task]
        for req in curr.get_requirements():
            if req not in order:
                order[req] = order[curr] + 1
                to_process.append(req)
            elif order[req] <= order[curr]:
                order[req] = order[curr] + 1
                to_process.append(req)
    return [task for task, _ in sorted(order.items(), key=lambda x: x[1], reverse=True)]


def execute_task(root_task: Type[Task]):
    order = linearize(root_task)
    for task in order:
        last_edited_dependency = None
        for dependency in task.get_requirements():
            dependency: Type[Task]
            last_modified = dependency.last_update_time()
            if last_modified is None:
                raise RuntimeError(f"task {task} depends on a non updated task {dependency}")
            if last_edited_dependency is None or last_edited_dependency < last_modified:
                last_edited_dependency = last_modified
        curr_output_modification_time = task.last_update_time()
        if curr_output_modification_time is None or \
                (last_edited_dependency is not None and last_edited_dependency > curr_output_modification_time):
            print(f"executing task {task}")
            task.execute()
            print(f"finished executing task {task}")
        else:
            print(f"task {task} is up to date")


def main():
    execute_task(TaskNormalizedMatrix)
    execute_task(TaskNormalizedMatrixNoIDF)
    execute_task(TaskExtractVectors)


if __name__ == "__main__":
    main()
