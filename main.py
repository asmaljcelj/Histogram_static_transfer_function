import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

limit_for_peaks = 19
colors_used = []


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
def create_histogram_demo(data):
    # demo
    ax = plt.axes(projection='3d')
    ranges = create_range_of_dimension(data)
    X, Y = np.meshgrid(ranges[0], ranges[1])
    ax.scatter3D(X, Y, data)
    ax.view_init(0, 35)
    plt.show()


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
    for p in data:
        color_peak_cube_and_floor(p, peaks)


def color_peak_cube_and_floor(p, peaks):
    if p.peak:
        p.color = [0, 0, 255]
    elif p.x == 254:
        # cube (green)
        p.color = [0, 255, 0]
    elif p.x == 255:
        # floor (yellow)
        p.color = [255, 255, 0]
    elif p.x == 0:
        # air (black)
        p.color = [255, 255, 255]
    else:
        # not floor, cube or peak, interpolate
        # todo: interpolating!!!
        p.color = [255, 0, 0]


# utility functions
def create_range_of_dimension(data):
    # assuming source image is squared (equal width and height)
    length = get_data_length(data)
    return range(length), range(length)


def get_data_length(data):
    return int(math.sqrt(len(data)))


def get_data_at_index(data, x, y):
    length = get_data_length(data)
    return data[x + y * length]


if __name__ == '__main__':
    image_src = './images/transferFunction.png'
    # prepare the image (open image, read pixels and convert rgb to grayscale value)
    data = open_image_and_prepare_pixels(image_src)
    # find peaks of data and color every pixel in accordance
    find_peaks_and_color_image(data)
    # save the colors of pixel in a new image
    save_image(data, 'object_test.png')
