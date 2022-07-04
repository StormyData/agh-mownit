#!/bin/python3
import collections
import enum
import json
import multiprocessing
import multiprocessing.connection
import pathlib
import select
import sys
import uuid
from typing import TextIO
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


class MsgType(enum.Enum):
    PAGE_UUID = 0
    NEXT_LINK = 1
    NEW_PROCESS = 2
    COMMAND = 3
    REQUEST_URL = 4
    EXIT = 5
    NEW_URL = 6


class CmdType(enum.Enum):
    STOP = 0
    SET_THRESHOLD = 1


def process(process_number: int, pipe: multiprocessing.connection.Connection):
    seq = 0
    while True:
        pipe.send((MsgType.REQUEST_URL, None))
        msg_type, page_url = pipe.recv()
        if msg_type == MsgType.EXIT:
            return
        page_uuid = uuid.uuid1(process_number, seq)

        page = requests.get(page_url).text
        with open(pathlib.Path("data/pages").joinpath(str(page_uuid)), "w") as f:
            f.write(page)

        pipe.send((MsgType.PAGE_UUID, (page_url, page_uuid)))

        soup = BeautifulSoup(page, 'html.parser')
        for link in soup.find_all('a'):
            path = link.get('href')
            if not path:
                continue
            if path.startswith('/'):
                path = urljoin(page_url, path)
            if not path.startswith("https://en.wikipedia.org"):
                continue

            pipe.send((MsgType.NEXT_LINK, path))
        seq += 1


def scheduler(master: multiprocessing.connection.Connection, threshold = float('inf')):
    connections: set[multiprocessing.connection.Connection] = {master}
    with open("data/checkpoint_set.json", "r") as f:
        url_set = set(json.load(f))
    with open("data/checkpoint_dict.json", "r") as f:
        url_dict = {key: uuid.UUID(value) for key, value in dict(json.load(f)).items()}

    waiting_queue = collections.deque()

    url_queue = collections.deque()
    url_queue.extend(url_set)

    do_exit = False
    while len(connections) > (1 if do_exit else 0):  # master should newer be removed
        if len(url_dict) > threshold:
            if not do_exit:
                print("threshold reached, exiting")
            do_exit = True
        r_list, _, _ = select.select(connections, [], [])
        for conn in r_list:
            conn: multiprocessing.connection.Connection
            msg_type, data = conn.recv()
            match msg_type:
                case MsgType.REQUEST_URL:
                    if do_exit:
                        conn.send((MsgType.EXIT, None))
                        connections.remove(conn)
                        print(f"processes left {len(connections) - 1}")
                    else:
                        if len(url_queue) > 0:
                            conn.send((MsgType.NEW_URL, url_queue.popleft()))
                        else:
                            waiting_queue.append(conn)
                case MsgType.NEXT_LINK:
                    if data not in url_dict and data not in url_set:
                        url_set.add(data)
                        if len(waiting_queue) > 0:
                            c = waiting_queue.popleft()
                            c.send((MsgType.NEW_URL, data))
                        else:
                            url_queue.append(data)
                case MsgType.PAGE_UUID:
                    page_url, page_uuid = data
                    url_set.remove(page_url)
                    url_dict[page_url] = page_uuid

                case MsgType.NEW_PROCESS:
                    connections.add(data)
                case MsgType.COMMAND:
                    cmd_type = data[0]
                    args = data[1:]
                    match cmd_type:
                        case CmdType.STOP:
                            print("stopping")
                            do_exit = True
                        case CmdType.SET_THRESHOLD:
                            threshold = args[0]

    with open("data/checkpoint_set.json", "w") as f:
        json.dump(list(url_set), f)
    with open("data/checkpoint_dict.json", "w") as f:
        json.dump({key: str(value) for key, value in url_dict.items()}, f)
    print("scheduler exiting")
    master.send((MsgType.EXIT, None))


def command_parser(conn: multiprocessing.connection.Connection):
    print(">", end="")
    do_exit = False
    while True:
        r_list, _, _ = select.select([conn] if do_exit else [conn, sys.stdin], [], [])
        for v in r_list:
            if v is sys.stdin:
                v: TextIO
                cmd = v.readline().split()
                match cmd:
                    case ["stop"]:
                        conn.send((MsgType.COMMAND, (CmdType.STOP,)))
                        do_exit = True
                    case ["set_threshold", val]:
                        conn.send((MsgType.COMMAND, (CmdType.SET_THRESHOLD, int(val))))
                    case _:
                        print("invalid command; valid commands: stop, set_threshold <int>")
                print(">", end="")
            if v is conn:
                v: multiprocessing.connection.Connection
                msg_type, arg = v.recv()
                if msg_type == MsgType.EXIT:
                    return


def main():
    threshold = 10000
    if not pathlib.Path("data/checkpoint_dict.json").exists():
        with open("data/checkpoint_dict.json", "w") as f:
            json.dump(dict(),f)
    if not pathlib.Path("data/checkpoint_set.json").exists():
        with open("data/checkpoint_set.json", "w") as f:
            json.dump([r"https://en.wikipedia.org/"], f)
    n_processes = 100
    master_conn1, master_conn2 = multiprocessing.Pipe(duplex=True)
    master_process = multiprocessing.Process(target=scheduler, args=(master_conn1,threshold))
    master_process.start()
    processes = []
    for pr_index in range(n_processes):
        conn1, conn2 = multiprocessing.Pipe(duplex=True)
        proc = multiprocessing.Process(target=process, args=(pr_index, conn2))
        proc.start()
        processes.append(proc)
        master_conn2.send((MsgType.NEW_PROCESS, conn1))

    command_parser(master_conn2)

    master_process.join()
    print("scheduler exited")
    for proc in processes:
        proc.join()
    print("exiting")


if __name__ == "__main__":
    main()
