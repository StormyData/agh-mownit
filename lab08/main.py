import networkx
# Directed graph (each unordered pair of nodes is saved once): web-Google.txt
# Webgraph from the Google programming contest, 2002
# Nodes: 875713 Edges: 5105039
# FromNodeId	ToNodeId
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import pickle


def extract_matrix():
    graph = dict()
    matrix = scipy.sparse.dok_matrix((875713, 875713), dtype="float")
    vertices = dict()
    with open("web-Google.txt") as f:
        for line in f.readlines():
            u, v = line.split()
            u, v = int(u), int(v)
            if u not in vertices:
                vertices[u] = len(vertices)
            u = vertices[u]
            if v not in vertices:
                vertices[v] = len(vertices)
            v = vertices[v]
            if u not in graph:
                graph[u] = dict()
            if v not in graph[u]:
                graph[u][v] = 0
            graph[u][v] += 1
    for u in graph:
        one_over_nu = 1.0/len(graph[u])
        for v in graph[u]:
            matrix[u, v] = one_over_nu
    with open("matrix.pickle", "wb") as f:
        pickle.dump(matrix, f)


def power_iteration(A: scipy.sparse.dok_matrix, n: int, epsilon: float):
    x = np.ones(A.shape[0])
    for i in range(n):
        dx = A.dot(x)
        factor = max(dx.max(), dx.min())
        dx = dx / factor
        if np.linalg.norm(x - dx) < epsilon:
            return factor, dx / np.linalg.norm(dx)
        x = dx
    return factor, x / np.linalg.norm(x)


def main():
    with open("matrix.pickle", "rb") as f:
        matrix = pickle.load(f)
    d = 1.0
    val, vec = power_iteration(d * matrix, 100, 0.0000000001)
    print(val, np.max(vec))


if __name__ == '__main__':
    main()

