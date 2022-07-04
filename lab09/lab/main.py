import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import numpy.fft

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def draw(img):
    plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.show()
def draw3d(img):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # x = []
    # y = []
    # z = []
    # for nx in range(img.shape[0]):
    #     for ny in range(img.shape[1]):
    #         z.append(img[nx, ny])
    #         x.append(nx)
    #         y.append(ny)
    x = np.arange(0, img.shape[1])
    y = np.arange(0, img.shape[0])
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, img)
    plt.show()

def zad1():
    img = mpimg.imread('Lab9_galia.png')
    img = rgb2gray(img)
    img = 1 - img
    dft = numpy.fft.fft2(img)
    phase = np.angle(dft)
    #phase /= 2 * numpy.pi
    #phase += 0.5
    modulus = np.absolute(dft)
    #modulus /= np.average(modulus)
    draw3d(phase)
    draw3d(modulus)


    #draw(phase)
    #draw(modulus)

def f(i1,i2,inv=False, th=0.9):
    img = mpimg.imread(i1)
    img = rgb2gray(img) / 255
    if inv:
        img = 1 - img
    dft = numpy.fft.fft2(img)
    img_e = mpimg.imread(i2)
    img_e = rgb2gray(img_e) / 255
    if inv:
        img_e = 1 - img_e
    dft_e = numpy.fft.fft2(np.rot90(img_e, k=2), s=img.shape)
    dft_2 = dft * dft_e
    img_2 = numpy.fft.ifft2(dft_2)
    img_2 = np.real(img_2)
    draw3d(img_2)
    max_v = np.max(img_2) * th
    draw3d(img_2 > max_v)
    draw((img_2 > max_v) + img)
    print(np.sum(img_2 > max_v))


def zad2():
    # f('Lab9_galia.png', 'Lab9_galia_e.png', inv=True, th =0.9)
    f('Lab9_school.jpg', 'Lab9_fish1.png', inv=False, th=0.7)



if __name__ == "__main__":
    zad2()