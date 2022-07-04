from scipy.optimize import dual_annealing

def arg_to_order(x):
    indexes = list(range(len(x)))
    out = []
    for i in x:
        curr = floor(i)
        if curr == len(indexes):
            curr-=1
        out.append(indexes.pop(curr))
    return out

#def fitness_function(x, points):
#    return total_distance(arg_to_order(x), points)

#
# def callback(x, f, context):
#     order = arg_to_order(x)
#     graph = networkx.DiGraph()
#     graph.name = f"value = {f}"
#     last = order[0]
#     for i in range(len(order)):
#         graph.add_node(i, pos=points[i])
#     for curr in order[1:]:
#         graph.add_edge(curr, last)
#         last = curr
#     graph.add_edge(last, order[0])
#     pos = [graph.nodes[e]['pos'] for e in graph.nodes]
#     networkx.draw_networkx(graph, pos)
#     plt.show()
def show(x, i):
    print(i, x)



from matplotlib.animation import FuncAnimation
def make_animation(hist, pos, *, skip=1, interval=10):
    hist = hist[::skip]
    n = len(hist)
    T_values = [T for T, v, x in hist]
    v_values = [v for T, v, x in hist]
    x_values = [x for T, v, x in hist]
    m = len(x_values[0])
    fig = plt.figure()
    def animate(frame):
        fig.clear()
        x = x_values[frame]
        graph = networkx.DiGraph()
        graph.add_nodes_from(range(m))
        graph.add_edges_from([(x[i - 1], x[i]) for i in range(m)])
        networkx.draw(graph, pos=pos, with_labels=True)
    ani = FuncAnimation(fig, animate, n, interval=interval, repeat=True)
    ani.save("aniamtion.mp4")
    #plt.show()





def render(frame):
    with open(f"/tmp/anim_temp/frame{frame:06d}.dot", "r") as f:
        graph = graphviz.Source(f.read(), directory="/tmp/anim_temp", filename=f"frame{frame:06d}.gv", engine="neato")
    graph.render(outfile=f"frame{frame:06d}.png")


def make_animation(hist, pos, *, skip=1, scale=0.05):
    try:
        hist = hist[::skip]
        n = len(hist)
        T_values = [T for T, v, x in hist]
        v_values = [v for T, v, x in hist]
        x_values = [x for T, v, x in hist]
        m = len(x_values[0])
        graph = graphviz.Digraph(directory="/tmp/anim_temp", engine="neato")
        for i in range(m):
            graph.node(str(i), str(i), pos=f"{pos[i][0] * scale},{pos[i][1] * scale}!")
        min_x, min_y = min(pos[i][0] for i in range(m)) * scale, min(pos[i][1] for i in range(m)) * scale
        print(f"rendering {n} frames")
        for frame in range(n):
            x = x_values[frame]
            dg = graph.copy()
            dg.node("title", f"T={T_values[frame]:.4f}\nv={v_values[frame]:.4f}", pos=f"{min_x},{min_y}!", shape="none")
            dg.edges([(str(x[i - 1]), str(x[i])) for i in range(m)])
            dg.save(f"frame{frame:06d}.dot")
            # dg.render(outfile=f"frame{frame:06d}.png")
        with multiprocessing.Pool(32) as p:
            p.map(render, range(n))
        # print(f"rendered frame {frame} out of {n}")
        print("finished, sticking")
        os.system("ffmpeg -r 30 -f image2 -i /tmp/anim_temp/frame%06d.png -y animation.mp4 2>/dev/null")
    finally:
        os.system("rm /tmp/anim_temp/*.png")
        os.system("rm /tmp/anim_temp/*.dot")
        os.system("rm /tmp/anim_temp/*.gv")
    print("done")

