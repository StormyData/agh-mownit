import os
import random
from math import ceil

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.special import expit

from annealing import do_annealing_stepped, make_graph_stepped, AnnealingCallback, AnnealingConfig


def show(mat: np.ndarray):
    plt.imshow(mat)
    plt.show()


def visualize(hist, skip=5):
    try:
        hist = hist[::skip]
        n = len(hist)
        for i in range(n):
            plt.imsave(f"/tmp/anim_temp/frame{i:06d}.png", hist[i][2])
        os.system("ffmpeg -r 30 -f image2 -i /tmp/anim_temp/frame%06d.png -y animation.mp4 2>/dev/null")
    finally:
        os.system("rm /tmp/anim_temp/*.png")


def visualize2(hist, skip=1, interval=1):
    hist = hist[::skip]
    n = len(hist)
    T_values = [T for T, v, x in hist]
    v_values = [v for T, v, x in hist]
    x_values = [x for T, v, x in hist]
    m = len(x_values[0])
    fig = plt.figure()

    def animate(frame):
        fig.clear()
        plt.imshow(x_values[frame])

    ani = FuncAnimation(fig, animate, n, interval=interval, repeat=True)
    # ani.save("animation.mp4")
    plt.show()


class BinImCallback(AnnealingCallback):
    def __init__(self, outfilename, skip: int = 5, framerate: int = 30):
        self.process = None
        self.outfilename = outfilename
        self.skip = skip
        self.framerate = framerate

    def on_setup(self):
        self.process = ffmpeg.input('pipe:', r=str(self.framerate), f='image2pipe') \
            .output(self.outfilename, vcodec='libx264') \
            .overwrite_output() \
            .run_async(pipe_stdin=True)

    def on_step(self, x, v: float, i: int):
        if i % self.skip == 0:
            plt.imsave(self.process.stdin, x)
        return False

    def on_cleanup(self):
        self.process.stdin.close()
        self.process.wait()


class BinImgConfig(AnnealingConfig):

    def __init__(self, T0: float, n: int, neighbourhood: [(int, int)], weights: [float]):
        super().__init__(T0, n)
        self.neighbourhood = neighbourhood
        self.weights = weights

    def energy(self, mat: np.ndarray):
        energies = np.zeros_like(mat, dtype="float")
        for shift, weight in zip(self.neighbourhood, self.weights):
            energies += (np.roll(mat, shift, axis=(0, 1))) * weight
        energies *= mat
        return -energies.sum()

    def swap_in_place(self, x, T: float):
        self.arbitrary_swap(x, T)

    def neighbour_swap(self, mat: np.ndarray, T):
        m = ceil(T)
        for i in range(m):
            x = random.randrange(0, mat.shape[0])
            y = random.randrange(0, mat.shape[1])
            dx, dy = random.choice(self.neighbourhood)
            ox, oy = (x + dx) % mat.shape[1], (y + dy) % mat.shape[1]
            mat[x, y], mat[ox, oy] = mat[ox, oy], mat[x, y]

    def arbitrary_swap(self, mat: np.ndarray, T):
        m = ceil(T)
        for i in range(m):
            x = random.randrange(0, mat.shape[0])
            y = random.randrange(0, mat.shape[1])
            ox = random.randrange(0, mat.shape[0])
            oy = random.randrange(0, mat.shape[1])
            mat[x, y], mat[ox, oy] = mat[ox, oy], mat[x, y]

    def schedule(self, T: float, i: int) -> float:
        return 0.997 * T

    def prob(self, v: float, v0: float, T: float) -> float:
        return 1 - expit((v - v0) / T)


# neighbourhood = [(-1, 0), (1, 0), (0, 1), (0, -1)]
# neighbourhood = [(-1, 0), (1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
# neighbourhood = [(0, -1), (0, 1)]
neighbourhood = [(-2, 0), (2, 0), (-2, 1), (2, 1), (-2, -1), (2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2)]
weights = [1/(x**2 + y**2)**0.5 for x, y in neighbourhood]
#weights = [1 for _ in neighbourhood]
p = 0.1
mat = np.random.choice((False, True), (256, 256), p=(1 - p, p))
out = do_annealing_stepped(mat, BinImgConfig(5000, 50000, neighbourhood, weights), BinImCallback("animation.mp4"))
print(out[0], out[1])
make_graph_stepped(out[2])
show(out[0])
# visualize2(out[2])
