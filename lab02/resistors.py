import matplotlib.pyplot as plt
import networkx
import numpy


def read_graph_list(graph_string: str):
    tokenized = graph_string.split()
    n = int(tokenized[0])
    e = int(tokenized[1])
    G = [[] for i in range(n)]
    E = []
    for edge in range(e):
        u, v, w = int(tokenized[3 * edge + 2]), int(tokenized[3 * edge + 3]), float(tokenized[3 * edge + 4])
        E.append((min(u, v), max(u, v), w))
        G[u].append(edge)
        G[v].append(edge)

    return G, E


def read_from_networkx(graph: networkx.Graph, weight: float):
    n = graph.number_of_nodes()
    E = []
    G = [[] for i in range(n)]
    nodes = {v:i for i, v in enumerate(graph.nodes)}
    for u, v in graph.edges:
        u = nodes[u]
        v = nodes[v]
        E.append((min(u, v), max(u, v), weight))
        e_n = len(E) - 1
        G[u].append(e_n)
        G[v].append(e_n)
    pos = networkx.spring_layout(graph)
    return G, E, {nodes[key]:pos[key] for key in pos}


def get_loops(E: list[(int, int, float)], source: (int, int, float)):
    E.append((min(source[0], source[1]), max(source[0], source[1]), 0))

    graph = networkx.Graph()
    edges = dict()
    base_edges = dict()
    cycles = []

    for edge_i in range(len(E)):
        u, v, _ = E[edge_i]
        if u == v:
            cycles.append([edge_i])
            continue
        if not (u, v) in edges:
            edges[(u, v)] = set()
            base_edges[(u, v)] = edge_i
        else:
            cycles.append([edge_i, base_edges[(u, v)]])
        edges[(u, v)].add(edge_i)

    graph.add_edges_from(base_edges.keys())
    basis = networkx.algorithms.cycle_basis(graph)
    for base in basis:
        cycle = []
        curr_vertex = base[0]
        for next_vertex in base[1:]:
            cycle.append(base_edges[(min(curr_vertex, next_vertex), max(next_vertex, curr_vertex))])
            curr_vertex = next_vertex
        cycle.append(base_edges[(min(curr_vertex, base[0]), max(base[0], curr_vertex))])
        cycles.append(cycle)
    E.pop()
    return cycles


def construct_equations_kirchoff(G: list[list[int]], E: list[(int, int, float)], source: (int, int, float)):
    e = len(E)
    n = len(G)
    equations = []
    indexes = [(i, "edge") for i in range(e)] + [(0, "source")]

    # 2 prawo
    for cycle in get_loops(E, source):
        eq = []
        off = 0
        if len(cycle) == 1:
            equations.append([(cycle[0], 1), 0])
            continue
        if cycle[1] == e:
            next_ = (source[0], source[1], 0)
        else:
            next_ = E[cycle[1]]

        if cycle[0] == e:
            if source[1] == next_[0] or source[1] == next_[0]:
                curr = source[0]
            else:
                curr = source[1]
        else:
            st = E[cycle[0]]
            if st[1] == next_[0] or st[1] == next_[1]:
                curr = st[0]
            else:
                curr = st[1]
        for edge in cycle:
            if edge == e:
                if curr == source[0]:
                    off = source[2]
                    curr = source[1]
                else:
                    off = -source[2]
                    curr = source[0]
                continue
            u, v, w = E[edge]
            if u == curr:
                eq.append((edge, w))
                curr = v
            else:
                eq.append((edge, -w))
                curr = u
        equations.append([eq, off])
    # 1 prawo
    for vertex_i in range(n):
        eq = []
        if vertex_i == source[0]:
            eq.append((e, -1))
        elif vertex_i == source[1]:
            eq.append((e, 1))
        for edge_i in G[vertex_i]:
            u, v, w = E[edge_i]
            if u == v:
                continue
            if u == vertex_i:
                eq.append((edge_i, -1))
            else:
                eq.append((edge_i, 1))

        equations.append([eq, 0])

    return equations, indexes


def construct_equations_potential(G: list[list[int]], E: list[(int, int, float)], source: (int, int, float)):
    e = len(E)
    n = len(G)
    equations = []
    equations.append([[(0, 1)], 0])  # bind 0-th node potential
    indexes = [(i, "vertex") for i in range(n)]
    for vertex_i in range(1, n):
        eq = []
        for edge_i in G[vertex_i]:
            u, v, w = E[edge_i]
            if u == v:
                continue
            if u == vertex_i:
                eq.append((v, -1 / w))
                eq.append((u, 1 / w))
            else:
                eq.append((v, 1 / w))
                eq.append((u, -1 / w))

        equations.append([eq, 0])

    equations[source[0]][0] += equations[source[1]][0]
    equations[source[0]][1] += equations[source[1]][1]
    eq = [(source[0], -1), (source[1], 1)]
    equations[source[1]] = (eq, source[2])

    return equations, indexes


def construct_matrix_from_equations(equations: [([(int, int)], int)]):
    h = len(equations)
    w = max(kp[0] for eq in equations for kp in eq[0]) + 1
    matrix = numpy.zeros(h * w).reshape((h, w))
    b = numpy.zeros(h)
    for eq_i in range(h):
        left, right = equations[eq_i]
        b[eq_i] += right
        for i, w in left:
            matrix[eq_i, i] += w
    if w == h:
        return matrix, b
    else:
        return matrix.T @ matrix, matrix.T @ b


def show_results(E: [(int, int, float)], source: (int, int, float), solved: [float], indexes: [(int, str)],labels = False, pos = None):
    def cmap(x, max_, min_):
        return plt.cm.plasma((x - min_) / (max_ - min_))

    graph = networkx.DiGraph()
    for line in range(len(solved)):
        sol = solved[line]
        index, of_type = indexes[line]
        if of_type == "vertex":
            graph.add_node(index, potential=sol)
        elif of_type == "edge":
            if sol > 0:
                graph.add_edge(E[index][0], E[index][1], current=sol, resistance=E[index][2], edgecolor=0)
            else:
                graph.add_edge(E[index][1], E[index][0], current=-sol, resistance=E[index][2], edgecolor=0)
    if pos is None:
        pos = networkx.spring_layout(graph, scale=1, k=5)
    node_labels = {e: f"{e}\nV={graph.nodes[e]['potential']:.2f}" for e in graph.nodes}
    edge_labels = {e: f"I={graph.edges[e]['current']:.2f}\nR={graph.edges[e]['resistance']:.2f}" for e in graph.edges}
    edge_colors = [graph.edges[e]['current'] for e in graph.edges]
    min_c = min(edge_colors)
    max_c = max(edge_colors)
    edge_colors = [cmap(c, min_c, max_c) for c in edge_colors]
    networkx.draw_networkx_nodes(graph, pos=pos, label=node_labels)
    if labels:
        networkx.draw_networkx_labels(graph, pos=pos, labels=node_labels)
    networkx.draw_networkx_edges(graph, pos=pos, label=edge_labels, edge_color=edge_colors)
    if labels:
        networkx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels)
    plt.title(f"voltage source between {source[0]} and {source[1]} with V={source[2]}")
    plt.show()


def add_missing_values(output: [float], indexes: [(int, str)], G: [[int]], E: [(int, int, float)], ground: int = None):
    output = list(output)
    n = len(G)
    e = len(E)
    included_edges = [-1] * e
    included_vertexes = [-1] * n

    for index_i in range(len(indexes)):
        index, of_type = indexes[index_i]
        if of_type == "edge":
            included_edges[index] = index_i
        elif of_type == "vertex":
            included_vertexes[index] = index_i

    for edge_i in range(e):
        if included_edges[edge_i] >= 0:
            continue
        u, v, w = E[edge_i]
        u_val = output[included_vertexes[u]]
        v_val = output[included_vertexes[v]]
        output.append((u_val - v_val) / w)
        indexes.append((edge_i, "edge"))
        included_edges[edge_i] = len(indexes) - 1

    if included_vertexes == [-1] * n:
        included_vertexes[0] = len(indexes)
        output.append(0)
        indexes.append((0, "vertex"))
    while min(included_vertexes) == -1:
        for vertex_i in range(n):
            if included_vertexes[vertex_i] >= 0:
                this_val = output[included_vertexes[vertex_i]]
                for edge_i in G[vertex_i]:
                    u, v, w = E[edge_i]
                    if u == vertex_i:
                        if included_vertexes[v] == -1:
                            included_vertexes[v] = len(indexes)
                            indexes.append((v, "vertex"))
                            I = output[included_edges[edge_i]]
                            output.append(this_val - I * w)
                    else:
                        if included_vertexes[u] == -1:
                            included_vertexes[u] = len(indexes)
                            indexes.append((u, "vertex"))
                            I = output[included_edges[edge_i]]
                            output.append(this_val + I * w)
    if ground is None:
        min_p = min(output[included_vertexes[vertex_i]] for vertex_i in range(n))
    else:
        min_p = output[included_vertexes[ground]]
    for vertex_i in range(n):
        output[included_vertexes[vertex_i]] -= min_p
    return output


def test(output, indexes, G, E, source, eps=0.00001):
    n = len(G)
    e = len(E)
    vertex_potentials = [0] * n
    edge_currents = [0] * e
    for index_i in range(len(indexes)):
        index, of_type = indexes[index_i]
        if of_type == "vertex":
            vertex_potentials[index] = output[index_i]
        elif of_type == "edge":
            edge_currents[index] = output[index_i]

    for vertex_i in range(n):
        if vertex_i == source[0] or vertex_i == source[1]:
            continue
        if abs(sum(edge_currents[edge_i] * (1 if vertex_i == E[edge_i][0] else -1) for edge_i in G[vertex_i])) > eps:
            raise ValueError(f"currents at node {vertex_i} do not sum to zero")
    s_currents = sum(edge_currents[edge_i] * (1 if source[0] == E[edge_i][0] else -1) for edge_i in G[source[0]])
    t_currents = sum(edge_currents[edge_i] * (1 if source[1] == E[edge_i][0] else -1) for edge_i in G[source[1]])
    if abs(s_currents + t_currents) > eps:
        raise ValueError(f"currents between start {source[0]} and end {source[1]} node do not sum to zero")
    for edge_i in range(e):
        u, v, w = E[edge_i]
        u_val = vertex_potentials[u]
        v_val = vertex_potentials[v]
        e_val = edge_currents[edge_i]
        if abs(u_val - v_val - e_val * w) > eps:
            raise ValueError(f"potentials across edge {edge_i} do not match with expected from current across")
    print("test passed")


def solve_and_test(G: list[list[int]], E: list[(int, int, float)], source: (int, int, float), method: str,labels: bool = False, pos=None):
    if method == "kirchoff":
        equation_generator = construct_equations_kirchoff
    elif method == "potential":
        equation_generator = construct_equations_potential
    else:
        raise ValueError("unknown solving method")
    equations, indexes = equation_generator(G, E, source)

    matrix, b = construct_matrix_from_equations(equations)

    solved = numpy.linalg.solve(matrix, b)
    solved = add_missing_values(solved, indexes, G, E)
    test(solved, indexes, G, E, source)
    show_results(E, source, solved, indexes, labels, pos)


graph = networkx.grid_2d_graph(10, 10)
G, E, pos = read_from_networkx(graph, 1)
source = (0, 99, 1)
solve_and_test(G, E, source, "kirchoff",False , pos)