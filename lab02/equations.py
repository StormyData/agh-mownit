import random
import timeit

import numpy as np


def gauss(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not isinstance(A, np.ndarray):
        raise ValueError("argument LU must be numpy.ndarray")
    if A.ndim != 2:
        raise ValueError("argument LU must be a 2 dimensional array")
    if A.shape[0] != A.shape[1]:
        raise ValueError("argument LU must be a square matrix")
    if not isinstance(b, np.ndarray):
        raise ValueError("argument b must be numpy.ndarray")
    if b.ndim != 1:
        raise ValueError("argument b must be a 1 dimensional array")
    if b.shape[0] != A.shape[0]:
        raise ValueError("argument b must be the same size as argument LU")
    A = A.copy()
    b = b.copy()
    n = A.shape[0]
    indexes = list(range(n))
    for i in range(n):
        factor = max(abs(A[i, :]))
        A[i, :] /= factor
        b[i] /= factor

    for i in range(n):
        maxv = -1
        maxx = -1
        maxy = -1
        for j in range(i, n):
            for k in range(i, n):
                if abs(A[j, k]) > maxv:
                    maxv = abs(A[j, k])
                    maxx = j
                    maxy = k

        if i != maxx:
            A[[i, maxx], :] = A[[maxx, i], :]
            b[i], b[maxx] = b[maxx], b[i]

        if i != maxy:
            A[:, [i, maxy]] = A[:, [maxy, i]]
            indexes[i], indexes[maxy] = indexes[maxy], indexes[i]

        if A[i, i] == 0:
            raise ValueError("matrix LU cannot be singular")
        for j in range(n):
            if i != j:
                factor = A[j, i] / A[i, i]
                A[j, :] -= factor * A[i, :]
                b[j] -= factor * b[i]

    return (b / A.diagonal())[indexes]


def factorLU(A: np.ndarray) -> np.ndarray:
    if not isinstance(A, np.ndarray):
        raise ValueError("argument LU must be numpy.ndarray")
    if A.ndim != 2:
        raise ValueError("argument LU must be a 2 dimensional array")
    if A.shape[0] != A.shape[1]:
        raise ValueError("argument LU must be a square matrix")
    A = A.copy()
    n = A.shape[0]
    for i in range(n):
        if A[i, i] == 0:
            raise ValueError("matrix LU cannot be singular")
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            A[j, i] = factor
    return A


def calcLU(LU: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not isinstance(LU, np.ndarray):
        raise ValueError("argument LU must be numpy.ndarray")
    if LU.ndim != 2:
        raise ValueError("argument LU must be a 2 dimensional array")
    if LU.shape[0] != LU.shape[1]:
        raise ValueError("argument LU must be a square matrix")
    if not isinstance(b, np.ndarray):
        raise ValueError("argument b must be numpy.ndarray")
    if b.ndim != 1:
        raise ValueError("argument b must be a 1 dimensional array")
    if b.shape[0] != LU.shape[0]:
        raise ValueError("argument b must be the same size as argument LU")
    n = LU.shape[0]
    x = b.copy()
    for i in range(1, n):
        x[i] -= np.dot(LU[i, :i], x[:i])
    for i in range(n - 1, -1, -1):
        x[i] = (x[i] - np.dot(LU[i, i + 1:], x[i + 1:])) / LU[i, i]
    return x


def normLU(A: np.ndarray, LU: np.ndarray) -> float:
    if not isinstance(LU, np.ndarray):
        raise ValueError("argument LU must be numpy.ndarray")
    if LU.ndim != 2:
        raise ValueError("argument LU must be a 2 dimensional array")
    if LU.shape[0] != LU.shape[1]:
        raise ValueError("argument LU must be a square matrix")
    if not isinstance(A, np.ndarray):
        raise ValueError("argument A must be numpy.ndarray")
    if LU.shape != A.shape:
        raise ValueError("argument A must be the same shape as LU")
    n = LU.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for x in range(1, n):
        for y in range(x):
            L[x, y] = LU[x, y]
    for i in range(n):
        L[i, i] = 1
    for x in range(n):
        for y in range(x,n):
            U[x, y] = LU[x, y]
    return np.linalg.norm(A - L @ U)


