import math
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.cm import get_cmap

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
    im = Image.new(mode="RGBA", size=(256, 256))
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
    color_map, interval = generate_color_map_and_interval_length('Blues', len(peaks), 0.1, 0.9)
    for i, p in enumerate(peaks):
        p.color = generate_peak_color(color_map, interval, 0.1, i)
    for p in data:
        color_pixels(p, peaks)


def color_pixels(p, peaks):
    if p.peak:
        return
    elif p.x == 254:
        # cube (green)
        p.color = [0, 255, 0, 255]
    elif p.x == 255:
        # floor (yellow)
        p.color = [255, 255, 0, 255]
    elif p.x == 0 or p.grayscale == 0:
        # air (black)
        p.color = [255, 255, 255, 0]
    else:
        # not floor, cube or peak, interpolate
        color = get_color_ratios(p, peaks)
        p.color = color


# utility functions
def get_data_length(data):
    return int(math.sqrt(len(data)))


def get_data_at_index(data, x, y):
    length = get_data_length(data)
    return data[x + y * length]


# generate color for peak
def generate_peak_color(color_map, interval, begin, i):
    position_in_interval = random.random() * interval
    position = i * interval + position_in_interval
    color = color_map(position + begin)
    return [int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 255]


# generate color map and interval length on custom interval from 0 to 1
def generate_color_map_and_interval_length(color_map_name, number_of_peaks, begin, end):
    cmap = get_cmap(color_map_name)
    interval = (end - begin) / number_of_peaks
    return cmap, interval


# multicolor gradient
def get_color_ratios(point, peaks):
    color_ratios = [1] * len(peaks)
    for index1, peak1 in enumerate(peaks):
        for index2, peak2 in enumerate(peaks):
            if index1 == index2:
                continue
            d = projection_distance(peak1, peak2, point)
            color_ratios[index1] *= limit(d)
    total_ratios_sum = 0
    for ratio in color_ratios:
        total_ratios_sum += ratio
    for ratio in color_ratios:
        ratio /= total_ratios_sum
    color = get_color_mix(peaks, color_ratios)
    return color


def get_color_mix(peaks, color_ratios):
    r = 0
    g = 0
    b = 0
    for index, peak in enumerate(peaks):
        r += peak.color[0] * color_ratios[index]
        g += peak.color[1] * color_ratios[index]
        b += peak.color[2] * color_ratios[index]
    return [int(r), int(g), int(b)]


def projection_distance(peak1, peak2, point):
    k2 = peak2.x * peak2.x - peak2.x * peak1.x + peak2.y * peak2.y - peak2.y * peak1.y
    k1 = peak1.x * peak1.x - peak2.x * peak1.x + peak1.y * peak1.y - peak2.y * peak1.y
    ab2 = (peak1.x - peak2.x) * (peak1.x - peak2.x) + (peak1.y - peak2.y) * (peak1.y - peak2.y)
    kcom = point.x * (peak1.x - peak2.x) + point.y * (peak1.y - peak2.y)
    d1 = (k1 - kcom) / ab2
    d2 = (k2 + kcom) / ab2
    return d2


def limit(value):
    if value < 0:
        return 0
    if value > 1:
        return 1
    return value


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
    save_image(data, 'object_test_alpha_with_shades_interpolate_test.png')
