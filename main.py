import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import ListedColormap

limit_for_peaks = 19

class Pixel:
    def __init__(self, x, y, color, peak, grayscale):
        self.x = x
        self.y = y
        self.color = color
        self.peak = peak
        self.grayscale = grayscale


# image processing
def convert_to_grayscale(pixel, index):
    x = index % 256
    y = int(index / 256)
    r = pixel[0]
    g = pixel[1]
    b = pixel[2]
    grayscale = 255 - (r + g + b) / 3
    return Pixel(x, y, [0, 0, 0], False, grayscale)


def open_image_and_prepare_pixels(src):
    im = Image.open(src, 'r')
    pixels = []
    pix_val = list(im.getdata())
    for i in range(len(pix_val)):
        pixels.append(convert_to_grayscale(pix_val[i], i))
    return pixels


def save_image(data, filename):
    im = Image.new(mode="RGB", size=(256, 256))
    for p in data:
        im.putpixel((p.x, p.y), tuple(p.color))
    im.save(filename)


# histogram
def find_peaks(data):
    length = get_data_length(data)
    peaks = []
    for p in data:
        if p.x == 0 or p.x == 254 or p.x == 255:
            continue
        if is_peak(p, data, length):
            peaks.append(p)
            p.peak = True
    return peaks


def is_peak(pixel, data, length):
    # look in all four directions and find if it is a peak
    # bottom
    if pixel.y < length - 1 and pixel.grayscale <= (
            get_data_at_index(data, pixel.x, pixel.y + 1).grayscale + limit_for_peaks):
        return False
    # right
    if pixel.x < length - 1 and pixel.grayscale <= (
            get_data_at_index(data, pixel.x + 1, pixel.y).grayscale + limit_for_peaks):
        return False
    # top
    if pixel.y > 1 and pixel.grayscale <= (
            get_data_at_index(data, pixel.x, pixel.y - 1).grayscale + limit_for_peaks):
        return False
    # left
    if pixel.x > 1 and pixel.grayscale <= (
            get_data_at_index(data, pixel.x - 1, pixel.y).grayscale + limit_for_peaks):
        return False
    # if all above is false, then we found a peak
    return True


def histogram(data):
    x = np.array(range(256))
    y = np.array(range(256))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(x, y, bins=(256, 256))
    xpos, ypos = np.meshgrid(yedges[:-1] + yedges[1:], xedges[:-1] + xedges[1:])

    xpos = xpos.flatten() / 2
    ypos = ypos.flatten() / 2

    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    dz = hist.flatten()

    rgba = find_peaks_and_color_image(data)

    ax.bar3d(xpos, ypos, data, dx, dy, dz, color=rgba)
    ax.view_init(elev=45, azim=45)
    plt.show()
    return data


def find_peaks_and_color_image(data):
    peaks = find_peaks(data)
    print("number of peaks: {}".format(len(peaks)))
    color_peaks(data, peaks)


def color_peaks(data, peaks):
    for p in peaks:
        p.color = generate_random_color()
    for p in data:
        color_peak_cube_and_floor(p, peaks)


def color_peak_cube_and_floor(p, peaks):
    if p.peak:
        return
    elif p.x == 254:
        # cube (green)
        p.color = [0, 255, 0]
    elif p.x == 255:
        # floor (yellow)
        p.color = [255, 255, 0]
    elif p.x == 0 or p.grayscale == 0:
        # air (black)
        p.color = [255, 255, 255]
    else:
        # not floor, cube or peak, interpolate
        # todo: interpolating!!!
        # p.color = [255, 0, 0]
        peak1, peak2 = find_closest_peaks(p, peaks)
        cmap = create_listed_colormap(peak1[0].color, peak2[0].color)
        percentage = calculate_percentage(peak1[1], peak2[1])
        color = cmap(percentage)
        p.color = [int(color[0]), int(color[1]), int(color[2])]


# utility functions
def get_data_length(data):
    return int(math.sqrt(len(data)))


def get_data_at_index(data, x, y):
    length = get_data_length(data)
    return data[x + y * length]


def create_listed_colormap(color1, color2):
    return ListedColormap([color1, color2])


def distance_between_pixels(p1, p2):
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)


def find_closest_peaks(pixel, peaks):
    min1 = ['', -1]
    min2 = ['', -1]
    for p in peaks:
        distance = distance_between_pixels(pixel, p)
        if distance < min1[1] or min1[1] == -1:
            min2 = min1
            min1 = [p, distance]
        elif distance < min2[1] or min2[1] == -1:
            min2 = [p, distance]
    return min1, min2


def generate_random_color():
    return list(np.random.choice(range(256), size=3))


def calculate_percentage(distance1, distance2):
    total_distance = distance1 + distance2
    return distance1 / total_distance


if __name__ == '__main__':
    image_src = './images/transferFunction.png'
    # prepare the image (open image, read pixels and convert rgb to grayscale value)
    data = open_image_and_prepare_pixels(image_src)
    # find peaks of data and color every pixel in accordance
    find_peaks_and_color_image(data)
    # save the colors of pixel in a new image
    save_image(data, 'object_test.png')
