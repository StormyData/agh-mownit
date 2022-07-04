import random

import numpy as np
import matplotlib.pyplot as plt


def gen_points(n, m):
    x = []
    y = []
    z = []
    step_n = 2 * np.pi / n
    step_m = np.pi / m
    for i in range(n):
        for j in range(m):
            ci = np.cos(i * step_n)
            si = np.sin(i * step_n)
            cj = np.cos(j * step_m)
            sj = np.sin(j * step_m)
            x.append(ci * sj)
            y.append(si * sj)
            z.append(cj)
    return x, y, z


def restore(s,u,vh,k):
    return sum(s[i] * np.outer(u[:, i], vh[i, :]) for i in range(k))

def draw2():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x, y, z = gen_points(100, 100)

    A = np.array([[random.randrange(-10, 10) for _ in range(3)] for _ in range(3)], dtype='float')
    M = np.array([x, y, z])
    u, s, vh = np.linalg.svd(A)
    s[0]= s[2]*100
    A2 = restore(s,u,vh,3)
    M2 = A2 @ M

    ax.scatter(xs=M2[0], ys=M2[1], zs=M2[2], s=0.1)
    for i in range(3):
        u_v = u[:, i]
        vh_v = vh[i, :]
        ax.plot3D(*[[0, s[i] * u_v[i]] for i in range(3)], c=["red","green","blue"][i])
        ax.plot3D(*[[0, s[i] * vh_v[i]] for i in range(3)], c=["red","green","blue"][i])

    plt.show()


def draw():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x, y, z = gen_points(100, 100)

    A = np.array([[random.randrange(-10, 10) for _ in range(3)] for _ in range(3)], dtype='float')
    M = np.array([x, y, z])
    M2 = A @ M
    u, s, vh = np.linalg.svd(A)

    ax.scatter(xs=M2[0], ys=M2[1], zs=M2[2], s=0.1)
    for i in range(3):
        u_v = u[:, i]
        vh_v = vh[i, :]
        ax.plot3D(*[[0, s[i] * u_v[i]] for i in range(3)], c=["red","green","blue"][i])
        ax.plot3D(*[[0, s[i] * vh_v[i]] for i in range(3)], c=["red","green","blue"][i])

    plt.show()
