import random

import ffmpeg
import graphviz
import matplotlib.pyplot as plt
import networkx
import numpy as np
import tempfile
from scipy.special import expit

from annealing import make_graph_stepped, AnnealingConfig, AnnealingCallback, do_annealing_stepped


def generate_points_uniform(n):
    return [(random.uniform(-100, 100), random.uniform(-100, 100)) for _ in range(n)]


def generate_points_9_groups(n):
    scale = 100 / 3
    means = [(3 * scale * x, 3 * scale * y) for x in range(3) for y in range(3)]
    points = np.concatenate([np.random.multivariate_normal(mean, [[scale, 0], [0, scale]], size=n) for mean in means],
                            axis=0)
    return points


def generate_points_normal(n, std):
    return np.random.multivariate_normal((0, 0), std, size=n)


class TSPConfig(AnnealingConfig):

    def __init__(self, T0: float, n: int, points: [(float, float)], swap_type=0):
        super().__init__(T0, n)
        self.swap_type = swap_type
        self.points = points
        self.n_cities = len(points)

    @staticmethod
    def distance(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def energy(self, order) -> float:
        return sum(TSPConfig.distance(self.points[order[i - 1]], self.points[order[i]]) for i in range(self.n_cities))

    def swap_in_place(self, x, T: float):
        match self.swap_type:
            case 0:
                self.arbitrary_swap(x, T)
            case 1:
                self.local_swap(x, T)

    def arbitrary_swap(self, indexes, T):
        a, b = random.sample(range(self.n_cities), k=2)
        indexes[a], indexes[b] = indexes[b], indexes[a]

    def local_swap(self, indexes, T):
        a = random.randrange(0, self.n_cities)
        indexes[a], indexes[a - 1] = indexes[a - 1], indexes[a]

    def schedule(self, T: float, i: int) -> float:
        return 0.997 * T

    def prob(self, v: float, v0: float, T: float) -> float:
        return 1 - expit((v - v0) / T)


class TSPCallback(AnnealingCallback):
    def __init__(self, outfilename,temp_dir, points: [(float, float)], scale: float = 0.05, skip: int = 5, framerate: int = 30):
        self.process = None
        self.outfilename = outfilename
        self.skip = skip
        self.framerate = framerate
        self.graph = graphviz.Digraph(directory=temp_dir, engine="neato")
        self.n_cities = len(points)
        for i in range(self.n_cities):
            self.graph.node(str(i), str(i), pos=f"{points[i][0] * scale},{points[i][1] * scale}!")
        self.min_x = min(points[i][0] for i in range(self.n_cities)) * scale
        self.min_y = min(points[i][1] for i in range(self.n_cities)) * scale

    def on_setup(self):
        self.process = ffmpeg.input('pipe:', r=str(self.framerate), f='image2pipe') \
            .output(self.outfilename, vcodec='libx264') \
            .overwrite_output() \
            .run_async(pipe_stdin=True)

    def on_step(self, x, v: float, i: int):
        if i % self.skip == 0:
            dg = self.graph.copy()
            dg.node("title", f"v={v:.4f}", pos=f"{self.min_x},{self.min_y}!", shape="none")
            dg.edges([(str(x[i - 1]), str(x[i])) for i in range(self.n_cities)])
            path = dg.render(outfile=f"frame{i // self.skip:06d}.png")
            with open(path, "rb") as f:
                data = f.read()
                self.process.stdin.write(data)
        return False

    def on_cleanup(self):
        self.process.stdin.close()
        self.process.wait()


def show2(x, pos):
    graph = networkx.DiGraph()
    m = len(x)
    graph.add_nodes_from(range(m))
    graph.add_edges_from([(x[i - 1], x[i]) for i in range(m)])
    networkx.draw_networkx(graph, pos)
    plt.show()


n = 5
points = generate_points_9_groups(n)
#points = generate_points_normal(20, [[0, 900], [900, 0]])
with tempfile.TemporaryDirectory() as tmpdir:
    out = do_annealing_stepped(list(range(len(points))), TSPConfig(5000, 6000, points),
                               TSPCallback("animation.mp4", tmpdir, points, skip=10))
print(out[0], out[1])
print(len(out[2]))
show2(out[0], points)
make_graph_stepped(out[2])
