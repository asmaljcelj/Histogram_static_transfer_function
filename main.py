import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


limit_for_peaks = 19


# image processing
def convert_to_grayscale(pixel):
    r = pixel[0]
    g = pixel[1]
    b = pixel[2]
    return 255 - (r + g + b) / 3


def open_image_and_prepare_pixels(src):
    im = Image.open(src, 'r')
    pix_val = list(im.getdata())
    for i in range(len(pix_val)):
        pix_val[i] = convert_to_grayscale(pix_val[i])
    return pix_val


def save_image(data, filename):
    im = Image.new(mode="RGB", size=(256, 256))
    for i in range(256):
        for j in range(256):
            im.putpixel((i, j), tuple(get_data_at_index(data, i, j)))
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


def histogram(data):
    h = np.histogram(data, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170,
                                 180, 190, 200, 210, 220, 230, 240, 250])


def find_peaks(data):
    length = get_data_length(data)
    peaks = []
    # traverse data and find peaks
    for i in range(length - 2):
        for j in range(length - 2):
            if i == 0 or j == 0:
                continue
            if is_peak(data, length, i, j):
                peaks.append([i, j])
    return peaks


def is_peak(data, length, x, y):
    # look in all four directions and find if it is a peak
    # bottom
    if y < length - 1 and get_data_at_index(data, x, y) <= (get_data_at_index(data, x, y + 1) + limit_for_peaks):
        return False
    # right
    if x < length - 1 and get_data_at_index(data, x, y) <= (get_data_at_index(data, x + 1, y) + limit_for_peaks):
        return False
    # top
    if y > 1 and get_data_at_index(data, x, y) <= (get_data_at_index(data, x, y - 1) + limit_for_peaks):
        return False
    # left
    if x > 1 and get_data_at_index(data, x, y) <= (get_data_at_index(data, x - 1, y) + limit_for_peaks):
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

    peaks = find_peaks(data)
    print("number of peaks: {}".format(len(peaks)))
    rgba = color_peaks(data, peaks)

    # ax.bar3d(xpos, ypos, data, dx, dy, dz, color=rgba)
    # ax.view_init(elev=45, azim=45)
    # plt.show()
    return rgba


def color_peaks(data, peaks):
    length = get_data_length(data)
    # rgba = []
    # traverse data and color peaks
    # for i in range(length - 2):
    #     for j in range(length - 2):
    #         if [i, j] in peaks:
    #             rgba.append([0, 0, 255])
    #         else:
    #             rgba.append([255, 0, 0])
    return [color_peak_cube_and_floor(i, peaks, length) for i in range(len(data))]


def color_peak_cube_and_floor(i, peaks, length):
    x = int(i % length)
    y = int(i / length)
    if [x, y] in peaks:
        return [0, 0, 255]
    elif x == 254:
        # cube (green)
        return [0, 255, 0]
    elif x == 255:
        # floor (yellow)
        return [255, 255, 0]
    return [255, 0, 0]



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
    # find peaks of data
    # peaks = find_peaks(data)
    # print(peaks)
    # print(len(peaks))
    color = histogram(data)

    save_image(color, 'result_limit_19_with_cube_floor.png')
