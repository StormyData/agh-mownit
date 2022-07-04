import os
import random
from math import ceil, exp

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.special import expit
import networkx as nx
import matplotlib as mpl
from annealing import do_annealing_stepped, make_graph_stepped, AnnealingCallback, AnnealingConfig

narr = []
for x in range(9):
    narr.append(([x for y in range(9)], [y for y in range(9)]))
for y in range(9):
    narr.append(([x for x in range(9)], [x for y in range(9)]))
for bx in range(3):
    for by in range(3):
        narr.append(
            ([3 * bx + x for x in range(3) for y in range(3)], [3 * by + y for x in range(3) for y in range(3)]))



def read_f(fname):
    with open(fname,"r") as f:
        return read_s(f.read())

def read_s(string):
    return np.array([[int(v) for v in line] for line in string.replace("x","0").split("\n")])
mask_str = """400600123
000002070
001030906
700928000
916000382
000361004
508010600
090800000
124006007"""
mask_str = """804090005
100728000
006034010
600200150
010000090
059007002
040970800
000382004
500010903"""
mask = read_s(mask_str)


class SudokuConfig(AnnealingConfig):
    def __init__(self, mask: np.ndarray, T0: float, n: int):
        super().__init__(T0, n)
        self.mask = mask

    def energy(self, mat: np.ndarray):
        val = 0
        for xs, ys in narr:
            for i in range(1, 10):
                v = sum(mat[xs, ys] == i)
                val += (v - 1 if v > 1 else 0)
        return val

    def swap_in_place(self, x, T: float):
        self.in_box_swap(x, T)

    def arbitrary_swap(self, mat: np.ndarray, T):
        pos1, pos2 = random.choices([(x, y) for x in range(9) for y in range(9) if self.mask[x, y] == 0], k=2)
        mat[pos1[0], pos1[1]], mat[pos2[0], pos2[1]] = mat[pos2[0], pos2[1]], mat[pos1[0], pos1[1]]

    def in_box_swap(self, mat: np.ndarray, T):
        available = []
        while len(available) < 2:
            bx, by = random.choice([(x, y) for x in range(3) for y in range(3)])
            available = [(3 * bx + x, 3 * by + y) for x in range(3) for y in range(3) if self.mask[x, y] == 0]
        pos1, pos2 = random.choices(available, k=2)
        mat[pos1[0], pos1[1]], mat[pos2[0], pos2[1]] = mat[pos2[0], pos2[1]], mat[pos1[0], pos1[1]]

    def schedule(self, T: float, i: int) -> float:
        return 0.995 * T

    def prob(self, v: float, v0:float, T: float) -> float:
        return 1 - expit((v - v0) / T)


class SudokuCallback(AnnealingCallback):
    def __init__(self, outfilename, framerate: int = 30):
        self.process = None
        self.outfilename = outfilename
        self.framerate = framerate

    def on_setup(self):
        self.process = ffmpeg.input('pipe:', r=str(self.framerate), f='image2pipe') \
            .output(self.outfilename, vcodec='libx264') \
            .overwrite_output() \
            .run_async(pipe_stdin=True)

    def on_step(self, mat, v, i):
        plt.imsave(self.process.stdin, mat)
        return v == 0

    def on_cleanup(self):
        self.process.stdin.close()
        self.process.wait()


def init(mask: np.ndarray):
    mat = mask.copy()
    for bx in range(3):
        for by in range(3):
            count = [1] * 10
            for x in range(3):
                for y in range(3):
                    count[mat[3 * bx + x, 3 * by + y]] -= 1
            to_fill = sum(([i] * count[i] for i in range(1, 10)), start=[])
            for x in range(3):
                for y in range(3):
                    if mat[3 * bx + x, 3 * by + y] == 0:
                        mat[3 * bx + x, 3 * by + y] = to_fill.pop()
    return mat


out = do_annealing_stepped(init(mask), SudokuConfig(mask, 500, 3000), SudokuCallback("animation.mp4"))
print(out[0], out[1], out[3])
make_graph_stepped(out[2])
