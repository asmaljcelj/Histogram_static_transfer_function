from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math


# image processing
def convert_to_grayscale(pixel):
    r = pixel[0]
    g = pixel[1]
    b = pixel[2]
    return (r + g + b) / 3


def open_image_and_prepare_pixels(src):
    im = Image.open(src, 'r')
    pix_val = list(im.getdata())
    for i in range(len(pix_val)):
        pix_val[i] = -convert_to_grayscale(pix_val[i])
    return pix_val


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
    if y < length - 1 and get_data_at_index(data, x, y) <= get_data_at_index(data, x, y + 1):
        return False
    # right
    if x < length - 1 and get_data_at_index(data, x, y) <= get_data_at_index(data, x + 1, y):
        return False
    # top
    if y > 1 and get_data_at_index(data, x, y) <= get_data_at_index(data, x, y - 1):
        return False
    # left
    if x > 1 and get_data_at_index(data, x, y) <= get_data_at_index(data, x - 1, y):
        return False
    # if all above is false, then we found a peak
    return True


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
    peaks = find_peaks(data)
    print(peaks)
    print(len(peaks))
