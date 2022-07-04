import numpy as np
from scipy import fft
import timeit
import matplotlib.pyplot as plt


def get_ft_matrix(n: int) -> np.ndarray:
    angle = -2 * np.pi * 1.0j / n
    omega = np.exp(angle)
    matrix = np.ndarray((n, n), dtype='complex')
    for i in range(n):
        for j in range(n):
            matrix[i, j] = omega ** (i * j)
    return matrix


def ft_matrix(vector: np.ndarray) -> np.ndarray:
    n = vector.size
    matrix = get_ft_matrix(n)
    return matrix @ vector


def is_power_of_two(x):
    return x and (not (x & (x - 1)))


def fast_ft(vector: np.ndarray) -> np.ndarray:
    n = vector.size
    if not is_power_of_two(n):
        raise ValueError("vector size must be an power of 2")
    if n < 2:
        return vector
    lx = fast_ft(vector[::2])
    sux = fast_ft(vector[1::2]) * np.exp(-2.0j * np.pi / n) ** np.arange(0, n//2, 1)
    return np.concatenate((lx + sux, lx - sux))


def ift_matrix(vector: np.ndarray) -> np.ndarray:
    n = vector.size
    matrix = get_ft_matrix(n)
    return (matrix @ vector.conjugate()).conjugate() / n


def time_it(func, n):
    vector = np.random.random(16)
    lt = timeit.timeit(lambda: func(vector), number=100 * n) / n / 100
    vector = np.random.random(256)
    mt = timeit.timeit(lambda: func(vector), number=10 * n) / n / 10
    vector = np.random.random(4096)
    gt = timeit.timeit(lambda: func(vector), number=n) / n
    return lt, mt, gt


def main_zad1():
    vector = np.random.random(16)
    print("vector", vector)
    ft_vector = ft_matrix(vector)
    ft_scipy = fft.fft(vector)
    print("ft_vector", ft_vector)
    print("div", ft_vector / ft_scipy)
    print("ft_scipy", ft_scipy)
    restored = ift_matrix(ft_vector)
    print("restored", restored)
    print(np.std(restored - vector))
    fft_vector = fast_ft(vector)
    print(np.std(fft_vector - ft_vector))

    print("scipy ft:", time_it(fft.fft, 100))
    print("recursive ft:", time_it(fast_ft, 10))
    print("matrix ft:", time_it(ft_matrix, 1))


def main_zad2():
    n = 1000
    vec = np.arange(0, n, 1)
    vec = 2.0 * np.pi / n * vec

    a_vec = np.zeros_like(vec)
    for i in range(5):
        a_vec += np.sin((2 * i + 1) * vec)
    a_vec_dft = fft.fft(a_vec)
    b_vec = np.empty_like(vec)
    for i in range(5):
        b_vec[i*n//5: (i+1)*n//5] = np.sin((2 * i + 1)*vec[:n//5])
    b_vec_dft = fft.fft(b_vec)
    ax1 = plt.subplot(3, 2, 3, title="a real")
    plt.plot(np.real(a_vec_dft))
    ax2 = plt.subplot(3, 2, 5, title="a imag")
    plt.plot(np.imag(a_vec_dft))
    ax3 = plt.subplot(3, 2, 4, title="b real")
    plt.plot(np.real(b_vec_dft))
    ax4 = plt.subplot(3, 2, 6, title="b imag")
    plt.plot(np.imag(b_vec_dft))
    plt.subplot(3, 2, 1, title="a")
    plt.plot(a_vec)
    plt.subplot(3, 2, 2, title="b")
    plt.plot(b_vec)

    plt.show()


if __name__ == "__main__":
    #main_zad1()
    main_zad2()
