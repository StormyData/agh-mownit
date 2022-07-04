import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt


def f2(x: np.ndarray):
    return x**5 * np.exp(-x) * np.sin(x)


def f4_2d(x: np.ndarray, y: np.ndarray):
    x = x.reshape((-1, 1))
    y = y.reshape((1, -1))
    sum_v = np.concatenate([x for _ in range(y.shape[1])], axis=1) + np.concatenate([y for _ in range(x.shape[0])], axis = 0)
    return 1.0 / (np.sqrt(sum_v) * (1 + sum_v))


def f4(x: np.ndarray, y: np.ndarray):
    sum_v = x + y
    return 1.0 / (np.sqrt(sum_v) * (1 + sum_v))

def f5(x, y):
    return x**2 + y**2

def road_distance(t: np.ndarray, v: np.ndarray):
    return np.sum((t[1:] - t[:-1]) * (v[1:] + v[:-1])) / 2


def simpson(x:np.ndarray, y: np.ndarray):
    sum = 0
    for i in range(1, x.size, 2):
        sum += (x[i + 1] - x[i - 1]) / 6 * (y[i - 1] + 4 * y[i] + y[i + 1])
    return sum

def zad1():
    t = np.arange(0, 100)
    v = np.random.random(100) * 10 - 4
    dists = np.empty_like(v)
    for i in range(v.size):
        dists[i] = road_distance(t[:i], v[:i])
    plt.plot(t, v)
    plt.plot(t, dists)
    plt.show()

def zad2():
    x = np.array([1, 2, 3, 4, 5])
    y = f2(x)
    print(scipy.integrate.simpson(y, x))
    print(simpson(x, y))
    print(scipy.integrate.quad(f2, 1, 5))


def zad3b(nx, ny):
    if nx % 2 == 0:
        nx += 1
    if ny % 2 == 0:
        ny += 1

    print(nx, ny)
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-3, 3, ny)
    print(road_distance(x, np.array([road_distance(y, np.array([f5(x[i], y[j])  for j in range(y.size)])) for i in range(x.size)])))


def zad3():
    print(scipy.integrate.dblquad(f4, 0, 1, 0, lambda x: 1 - x))
    print(scipy.integrate.dblquad(f5, -3, 3, -5, 5))
    zad3b(1, 1)
    zad3b(3, 3)
    zad3b(5, 5)
    zad3b(10, 10)
    zad3b(100, 100)
    zad3b(500, 500)
    zad3b(1000, 1000)

if __name__ == "__main__":
    zad3()