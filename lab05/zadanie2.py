import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


img = mpimg.imread('lenna.png')
gray = rgb2gray(img)
u,s,vh= numpy.linalg.svd(gray)
r = np.linalg.matrix_rank(gray)
print(r)


def draw(k):
    gray2 = sum(s[i] * np.outer(u[:, i],vh[i, :]) for i in range(k))
    plt.imshow(gray2, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.show()

def f(k):
    gray2 = sum(s[i] * np.outer(u[:,i],vh[i,:]) for i in range(k))
    return np.linalg.norm(gray2 - gray)

def draw_plot():
    x = [i for i in range(r)]
    y = [f(i) for i in x]
    plt.plot(x,y)
    plt.show()

draw(10)