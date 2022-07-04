import random

from matplotlib import pyplot as plt
import tqdm


class AnnealingCallback:
    def on_step(self, x, v: float, i: int) -> bool:
        pass

    def on_cleanup(self) -> None:
        pass

    def on_finish(self) -> None:
        pass

    def on_setup(self) -> None:
        pass


class AnnealingConfig:
    def __init__(self, T0: float, n: int):
        self.T0 = T0
        self.n = n

    def energy(self, x) -> float:
        pass

    def swap_in_place(self, x, T: float):
        pass

    def schedule(self, T: float, i: int) -> float:
        pass

    def prob(self, v: float, v0: float, T: float) -> float:
        pass


# def do_annealing(x0, f, n, swap, T0, shedule, P):
#     x_best = x0
#     T = T0
#     v_best = f(x0)
#     hist = [(T, v_best, x_best)]
#     for i in tqdm.trange(n):
#         x_next = swap(x_best, T)
#         v_next = f(x_next)
#         if v_next < v_best or random.random() <= P(v_next, v_best, T):
#             x_best = x_next
#             v_best = v_next
#         hist.append((T, v_best, x_best))
#         T = shedule(T)
#     return x_best, v_best, hist


def do_annealing_stepped(x0, config: AnnealingConfig, callback: AnnealingCallback):
    try:
        callback.on_setup()
        x_best = x0
        x_curr = x0.copy()
        T = config.T0
        v_best = config.energy(x0)
        v_curr = v_best
        hist = [(T, v_curr)]
        for i in tqdm.trange(config.n):
            x_next = x_curr.copy()
            config.swap_in_place(x_next, T)
            v_next = config.energy(x_next)
            if v_next < v_curr or random.random() <= config.prob(v_next, v_curr, T):
                x_curr = x_next
                v_curr = v_next

            if v_curr < v_best:
                v_best = v_curr
                x_best = x_curr

            hist.append((T, v_curr))
            if callback.on_step(x_curr, v_curr, i):
                return x_best, v_best, hist, i
            T = config.schedule(T, i)
        callback.on_finish()
        return x_best, v_best, hist, config.n
    finally:
        callback.on_cleanup()


# def make_graph(hist):
#     T_values = [T for T, v, x in hist]
#     v_values = [v for T, v, x in hist]
#     plt.subplot(2, 1, 1)
#     plt.plot(T_values)
#     plt.subplot(2, 1, 2)
#     plt.plot(v_values)
#     plt.show()
#

def make_graph_stepped(hist):
    T_values = [T for T, v in hist]
    v_values = [v for T, v in hist]
    plt.subplot(2, 1, 1)
    plt.plot(T_values)
    plt.subplot(2, 1, 2)
    plt.plot(v_values)
    plt.show()