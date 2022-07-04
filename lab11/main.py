import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt

def myQR(A: np.ndarray):
    n = A.shape[0]
    Q = A.copy()
    R = np.zeros_like(A)
    for k in range(n):
        for i in range(k):
            Q[:, k] -= Q[:, i] * np.dot(Q[:, i], A[:, k])
        Q[:, k] /= np.linalg.norm(Q[:, k])
    for k in range(n):
        for i in range(k + 1):
            R[i, k] = np.dot(Q[:, i], A[:, k])
    return Q, R


def test2(n: int):
    x = np.random.random((n, n))
    Q, R = myQR(x)
    print(n, np.allclose(Q @ R, x))


def test3():
    def tmp():
        x = np.random.random((8, 8))
        u, s, vh = scipy.linalg.svd(x)
        s[0] *= 1000
        x = u @ np.diag(s) @ vh
        Q, R = myQR(x)
        return np.linalg.cond(x), np.linalg.norm(np.eye(8) - Q.T @ Q)

    l = [tmp() for i in range(5000)]
    l.sort(key=lambda x: x[0])
    x, y = list(zip(*l))
    plt.plot(x, y)
    plt.show()


def zad2(x, y):
    A = np.zeros((x.size, 3))
    A[:, 0] = np.ones_like(x)
    A[:, 1] = x
    A[:, 2] = x**2
    b = y

    Q, R = np.linalg.qr(A)
    x = b @ Q
    # R @ x = b
    n = x.size
    for i in range(n - 1, -1, -1):
        x[i] = (x[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]
    return x

def main():
    print(zad2(np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
                  np.array([2, 7, 9, 12, 13, 14, 14, 13, 10, 8, 4])))
    # print(myQR(np.array([[1,2,3],[4,5,6], [7,8,9]], dtype='double')))


if __name__ == "__main__":
    main()
