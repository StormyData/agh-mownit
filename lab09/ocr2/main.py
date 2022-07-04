import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy.linalg
import numpy.fft
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import portion
import scipy
import scipy.ndimage
import multiprocessing


def fast_hough_line(img, angle_step=1, lines_are_white=True, value_threshold=0.5):
    """hough line using vectorized numpy operations,
    may take more memory, but takes much less time"""
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))  # can be changed
    # width, height = col.size  #if we use pillow
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))  # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    # are_edges = cv2.Canny(img,50,150,apertureSize = 3)
    y_idxs, x_idxs = np.nonzero(are_edges)  # (row, col) indexes to edges
    # Vote in the hough accumulator
    xcosthetas = np.dot(x_idxs.reshape((-1, 1)), cos_theta.reshape((1, -1)))
    ysinthetas = np.dot(y_idxs.reshape((-1, 1)), sin_theta.reshape((1, -1)))
    rhosmat = np.round(xcosthetas + ysinthetas) + diag_len
    rhosmat = rhosmat.astype(np.int16)
    for i in range(num_thetas):
        rhos, counts = np.unique(rhosmat[:, i], return_counts=True)
        accumulator[rhos, i] = counts
    return accumulator, thetas, rhos


def get_angle(img):
    accumulator, thetas, rhos = fast_hough_line(img)
    idx = np.argmax(accumulator)
    theta = thetas[idx % accumulator.shape[1]]
    return 90 - theta / np.pi * 180


def svd_filter(arr: np.ndarray, percent: float = 0.90):
    u, s, vh = numpy.linalg.svd(arr)
    n_left = int(s.size * percent)
    s[n_left + 1:] = 0
    smat = np.zeros((u.shape[0], vh.shape[0]))
    smat[:s.size, :s.size] = np.diag(s)
    return u @ smat @ vh


def remove_replace_and_return_ith_element(arr: [], index: int):
    retval = arr[index]
    if index == len(arr) - 1:
        arr.pop()
    else:
        arr[index] = arr.pop()
    return retval


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def text_phantom(text, size, imsize, angle=None) -> np.ndarray:
    # Availability is platform dependent
    font = '/usr/share/fonts/noto/NotoSans-Regular.ttf'

    # Create font
    pil_font = ImageFont.truetype(font, size=size,
                                  encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', imsize, (255, 255, 255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    white = "#000000"
    draw.text((5, 5), text, font=pil_font, fill=white)
    if angle is not None:
        canvas = canvas.rotate(angle, fillcolor=(255, 255, 255), expand=True)
    # Convert the canvas into an array with values in [0, 1]
    return (255 - np.asarray(canvas)) / 255.0


pattern_text = "abcdefghijklmnoprstuwxyz0123456789.,?!"

detection_threshold = {k: 0.9 for k in pattern_text}
detection_threshold['i'] = 0.95
detection_threshold['l'] = 0.95
detection_threshold['.'] = 0.95
detection_threshold[','] = 0.95

exclusion_threshold = {k: 0.1 for k in pattern_text}
exclusion_threshold['r'] = 0.05
exclusion_threshold['.'] = 0.05
exclusion_threshold[','] = 0.05
exclusion_threshold['!'] = 0.05

precedeed = {k: set() for k in pattern_text}
# precedeed['o'] = {"p", "d", "b", "g"}
# precedeed['n'] = {'m', "p", 'b', 'h'}
# precedeed['c'] = set("eopb")
# precedeed['.'] = set("abcdefghijklmnoprstuwxyz0123456789,?!")
# precedeed[','] = set("abcdefghijklmnoprstuwxyz0123456789?!")
# precedeed['i'] = {'p', "l", "t"} #### <- doesn't detect
# precedeed['j'] = {'d', 'g'}
# precedeed['l'] = set("p1bdfghjk") #### <- invalid detection
# precedeed['r'] = {"m", "n", "p", "b", "h"} ### <- detected in m anyway
# precedeed['!'] = set("l1dghjt4bfkp")
# precedeed['u'] = {'d', 'g'}
# precedeed['h'] = {'b'}
# precedeed['1'] = {'4'}
# precedeed['3'] = {'8'}
# precedeed['?'] = {'7'}

precedeed['c'] = set('bdegop')
precedeed['h'] = set('b')
precedeed['i'] = set('bdfghjklpt14')
precedeed['j'] = set('dg')
precedeed['l'] = set('bdfghjkp14')
precedeed['n'] = set('bhmp')
precedeed['o'] = set('bdgp')
precedeed['r'] = set('bhmnp')
precedeed['u'] = set('dg')
precedeed['1'] = set('4')
precedeed['3'] = set('8')
precedeed['.'] = set('abdefghkmnprtuwxyz1246789?!')
precedeed[','] = set('abcdefghjkmnoprtuwxyz02345679')
precedeed['?'] = set('7')
precedeed['!'] = set('bdfghjklpt14')

letters_to_draw = "ilnr.,!"

svd_percent = 0.9


def gen_patterns():
    patterns = dict()
    master_img = rgb2gray(text_phantom(pattern_text, 100, (2000, 300)))
    master_img = svd_filter(master_img, svd_percent)
    master_img_dft = numpy.fft.fft2(master_img)
    for char in pattern_text:
        graphics = rgb2gray(text_phantom(char, 100, (300, 300)))
        graphics = graphics[:, np.any(graphics > 0, axis=0)]
        graphics = graphics[np.any(graphics > 0, axis=1), :]
        graphics = svd_filter(graphics, svd_percent)
        patterns[char] = (graphics, np.max(np.real(convolve_d(master_img_dft, graphics))))

    img = rgb2gray(text_phantom(pattern_text[0] + " " + pattern_text[0], 100, (150, 125)))
    img = svd_filter(img, svd_percent)
    boxes = get_positions(np.fft.fft2(img), {pattern_text[0]: patterns[pattern_text[0]]})
    boxes = boxes[pattern_text[0]]
    boxes.sort(key=lambda x: x[1] - x[3])
    zero_space_zero_dist = boxes[1][1] - boxes[1][3] - boxes[0][1]

    img = rgb2gray(text_phantom(pattern_text[0] + "  " + pattern_text[0], 100, (200, 125)))
    img = svd_filter(img, svd_percent)
    boxes = get_positions(np.fft.fft2(img), {pattern_text[0]: patterns[pattern_text[0]]})
    boxes = boxes[pattern_text[0]]
    boxes.sort(key=lambda x: x[1] - x[3])
    zero__space_space_zero_dist = boxes[1][1] - boxes[1][3] - boxes[0][1]

    space_width = zero__space_space_zero_dist - zero_space_zero_dist

    space_dists = dict()
    pairs = [(char_1, char_2) for char_1 in pattern_text for char_2 in pattern_text]
    with multiprocessing.Pool() as p:
        values = p.starmap(generate_pair_width, [(pair, patterns, space_width) for pair in pairs])
    for value, pair in zip(values, pairs):
        space_dists["".join(pair)] = value

    return patterns, space_dists, space_width


def generate_pair_width(pair, patterns, space_width):
    char_1, char_2 = pair
    img = rgb2gray(text_phantom(char_1 + " " + char_2, 100, (200, 125)))
    img = svd_filter(img, svd_percent)
    boxes = get_positions(np.fft.fft2(img), {char_1: patterns[char_1], char_2: patterns[char_2]})
    remove_exclusive_overlapping_boxes(boxes)
    for v in boxes.values():
        v.sort(key=lambda x: x[1] - x[3])
    if char_1 not in boxes or char_2 not in boxes or (
            char_1 != char_2 and (len(boxes[char_1]) != 1 or len(boxes[char_2]) != 1)) or (
            char_1 == char_2 and len(boxes[char_1]) != 2):
        # print(char_1, char_2, boxes)
        val = float('inf')
    else:
        left_end = boxes[char_1][0][1] if char_1 != char_2 else boxes[char_1][1][1]
        right_begin = boxes[char_2][0][1] - boxes[char_2][0][3]
        val = right_begin - left_end - space_width
    return val


def convolve(img: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    img_dft = numpy.fft.fft2(img)
    pattern_dft = numpy.fft.fft2(np.rot90(pattern, k=2), s=img.shape)
    result_dft = img_dft * pattern_dft
    return numpy.fft.ifft2(result_dft)


def convolve_d(img_dft: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    pattern_dft = numpy.fft.fft2(np.rot90(pattern, k=2), s=img_dft.shape)
    result_dft = img_dft * pattern_dft
    return numpy.fft.ifft2(result_dft)


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


def draw_onto_img_boxes_with_colors(img: np.ndarray, combined_boxes: dict[str, (int, int, int, int)],
                                    colored_chars: dict[str, (float, float, float)]):
    for char, color in colored_chars.items():
        if char in combined_boxes:
            for x, y, lx, ly in combined_boxes[char]:
                img[x - lx: x, y - ly: y] = np.array(color)


def draw(img):
    plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.show()


def drawRGB(img):
    plt.imshow(img, vmin=0, vmax=1)
    plt.show()


def merge_boxes(*boxes):
    x_min = min(x - lx for x, y, lx, ly in boxes)
    y_min = min(y - ly for x, y, lx, ly in boxes)
    x_max = max(x for x, y, lx, ly in boxes)
    y_max = max(y for x, y, lx, ly in boxes)
    return x_max, y_max, x_max - x_min, y_max - y_min


def get_box_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x3_l = max(x1 - w1, x2 - w2)
    x3_h = min(x1, x2)

    y3_l = max(y1, y2)
    y3_h = min(y1 + h1, y2 + h2)
    if x3_l < x3_h and y3_l < y3_h:
        area = (x3_h - x3_l) * (y3_h - y3_l)
        max_area = min(w1 * h1, w2 * h2)
        return area / max_area
    return 0.0


def store_patterns(patterns):
    with open("patterns.pickle", "wb") as f:
        return pickle.dump(patterns, f)


def load_patterns():
    with open("patterns.pickle", "rb") as f:
        return pickle.load(f)


def rotate_back(img, angle):
    # image = Image.fromarray(255 * img, mode="RGB")
    # image.show()
    # image = image.rotate(-angle, fillcolor=(0, 0, 0), expand=True)
    # return np.array(image) / 255.0
    return scipy.ndimage.rotate(img, -angle, reshape=True)


def main(img: np.ndarray):
    patterns, space_dists, space_width = load_patterns()
    print("getting angle")
    angle = get_angle(rgb2gray(img))

    if not np.isclose(angle, 180):
        print("rotating", -angle)
        img = rotate_back(img, angle)
    print("filtering")
    filtered = svd_filter(rgb2gray(img), svd_percent)
    print("fft2")
    img_dft = numpy.fft.fft2(filtered)
    print("getting positions")
    combined_boxes = get_positions(img_dft, patterns)
    # print(combined_boxes)
    print("mutually excluding")
    remove_exclusive_overlapping_boxes(combined_boxes)

    print("flattening")
    flattened_boxes = [(char, box) for char in combined_boxes for box in combined_boxes[char]]

    y_ranges = [portion.closed(box[0] - box[2], box[0]) for _, box in flattened_boxes]
    merged_y_ranges = portion.Interval(*y_ranges)

    line_intervals = [portion.closed(i.lower, i.upper) for i in merged_y_ranges]
    line_separated_boxes = dict()
    print("sorting into lines")
    for char, box in flattened_boxes:
        y_int = portion.closed(box[0] - box[2], box[0])
        for i, interval in enumerate(line_intervals):
            if interval.overlaps(y_int):
                if i not in line_separated_boxes:
                    line_separated_boxes[i] = []
                line_separated_boxes[i].append((char, box))
    lines = dict()
    space_width *= 0.9
    print("calculating whitespaces")
    for key, val in line_separated_boxes.items():
        val.sort(key=lambda x: x[1][1] - x[1][3])
        line = ""

        tmp = list(
            filter(lambda x: x[0] not in ".,lr!i", val))  # filtering out characters, that cannot be properly recognized

        for box_left, box_right in zip(tmp, tmp[1:]):
            line += box_left[0]
            left_end = box_left[1][1]
            right_begin = box_right[1][1] - box_right[1][3]
            offset = space_dists[box_left[0] + box_right[0]]
            if offset == float('inf'):
                offset = 0
            dist = right_begin - left_end - offset
            while dist > space_width:
                line += " "
                dist -= space_width
        line += tmp[-1][0]
        lines[key] = line
    return "\n".join(v for _, v in sorted(lines.items(), key=lambda x: x[0]))


def remove_exclusive_overlapping_boxes(combined_boxes):
    for char_1 in combined_boxes:
        for char_2 in combined_boxes:
            if char_2 not in precedeed[char_1]:
                continue
            i = 0
            cb = combined_boxes[char_1]
            while i < len(cb):
                for box_2 in combined_boxes[char_2]:
                    if get_box_overlap(cb[i], box_2) > exclusion_threshold[char_1]:
                        if i == len(cb) - 1:
                            cb.pop()
                        else:
                            cb[i] = cb.pop()
                        break
                else:
                    i += 1
    for char in set(combined_boxes.keys()):
        if len(combined_boxes[char]) == 0:
            combined_boxes.pop(char)


def get_positions(img_dft, patterns):
    combined_boxes = {}

    for char in patterns:
        boxes = []
        pattern, threshold = patterns[char]
        conv = convolve_d(img_dft, pattern)
        conv = np.real(conv)

        lx, ly = pattern.shape
        arr = np.argwhere(conv > detection_threshold[char] * threshold)
        for x, y in arr:
            boxes.append((x, y, lx, ly))

        combined = []
        for box in boxes:
            for i in range(len(combined)):
                if get_box_overlap(box, combined[i]) > 0.0:
                    combined[i] = merge_boxes(combined[i], box)
                    break
            else:
                combined.append(box)
        boxes = combined

        if len(boxes) > 0:
            combined_boxes[char] = boxes
    return combined_boxes


if __name__ == "__main__":
    if not pathlib.Path("patterns.pickle").exists():
        patterns = gen_patterns()
        store_patterns(patterns)
    txt = "\n".join([" ".join("abcdefghijklmnoprstuwxyz0123456789.,?!")] * 2)
    # img = text_phantom(txt, 100, [100 * len(txt), 300])
    # txt = # "lorem ipsum 123 !!! ?"
    img = text_phantom(txt, 100, (3000, 250), angle=10)
    draw(img)
    ans = main(img)
    print("returned\n" + "-" * 30)
    print(ans)
    print("-" * 30)
