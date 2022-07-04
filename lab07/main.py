import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.linalg import lu_factor, lu_solve
import scipy.linalg


def power_iteration(A: np.ndarray, n: int, epsilon: float):
    x = np.ones(A.shape[0])
    for i in range(n):
        dx = A @ x
        factor = np.max(np.abs(dx))
        dx = dx / factor
        if np.linalg.norm(x - dx) < epsilon:
            return factor, dx / np.linalg.norm(dx)
        x = dx
    return factor, x / np.linalg.norm(x)


def inverse_power_iteration(A: np.ndarray, n: int, epsilon: float, sigma: float):
    LU, pivot = lu_factor(A - np.eye(A.shape[0]) * sigma)
    x = np.ones(A.shape[0])
    for i in range(n):
        dx = lu_solve((LU, pivot), x)
        factor = np.max(np.abs(dx))
        dx = dx / factor
        if np.linalg.norm(x - dx) < epsilon:
            return sigma + 1/factor, dx / np.linalg.norm(dx)
        x = dx
    return sigma + 1/factor, x / np.linalg.norm(x)


def rayleigh_inverse_power_iteration(A: np.ndarray, n: int, epsilon: float, sigma: float):
    eigenval = sigma
    x = np.ones(A.shape[0])
    for i in range(n):
        dx = np.linalg.solve(A - np.eye(A.shape[0]) * eigenval, x)
        factor = np.max(np.abs(dx))
        dx = dx / factor
        eigenval = (x @ A @ x.T) / (x @ x.T)
        if np.linalg.norm(x - dx) < epsilon:
            return eigenval, dx / np.linalg.norm(dx)
        x = dx
    return eigenval, x / np.linalg.norm(x)




def main():
    A = np.array([[-149, -50, -154], [537, 180, 546], [-27, -9, -25]], dtype="float")
    for i in range(3):
        A[i, i:] = A[i:, i]
    #print(power_iteration(A, 10000, 0.00001))
    #print(inverse_power_iteration(A, 10000, 0.00001, 580))
    #print(inverse_power_iteration(A, 10000, 0.00001, -540))
    #print(inverse_power_iteration(A, 10000, 0.00001, -25))
    print(np.linalg.eig(A))
    print(rayleigh_inverse_power_iteration(A, 10000, 0.00001, 600))

    sizes = [10*i+10 for i in range(200)]
    times = []
    for size in sizes:
        A = np.random.random([size, size])
        for i in range(size):
            A[i, i:] = A[i:, i]
        times.append((timeit.timeit(lambda: power_iteration(A, 10000, 0.0000001),number=10)))
    plt.plot(sizes, times)
    plt.show()

main()
