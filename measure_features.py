#!/usr/bin/env python3

import sys
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.image as mimg
import matplotlib.path as mpath
from itertools import product, compress
from IPython import embed


__author__ = "jpresern", "bpiskur"


def prepare_storage():
    """
    sets up empty dataframe, containing following columns:
        sample      - name of photo
        parallel    - iteration on the same sample
        type        - what are we measuring (area, points, distance)
        x, y        - coordinates
        quality     - quality measured (surface if area, density of points inside the area if points, distance if distance)
        quantity    - measured value for quality
    :return df: pandas dataframe, ready
    """

    columns = ['sample', 'parallel', 'type', 'element', 'x', 'y', 'quality', 'quantity']
    df = pd.DataFrame(columns=columns)

    return df


def store_area(df, fn, xy_pairs, area, parallel=1):
    columns = ['sample', 'parallel', 'type', 'element', 'x', 'y', 'quality', 'quantity']
    df_empty = pd.DataFrame(columns=columns)
    for i, v in enumerate(xy_pairs):
        df_empty.loc[i, :] = fn, parallel, 'area', 'corner', v[0], v[1], 'surface', area
    return pd.concat([df, df_empty])


def store_features(df, fn, xy_pairs, area, parallel=1):
    columns = ['sample', 'parallel', 'type', 'element', 'x', 'y', 'quality', 'quantity']
    df_empty = pd.DataFrame(columns=columns)
    density = xy_pairs.shape[0]/area
    for i, v in enumerate(xy_pairs):
        df_empty.loc[i, :] = fn, parallel, 'area', 'points', v[0], v[1], 'density', density
    return pd.concat([df, df_empty])


def drawing_board():
    """ create image with axes """
    figure = plt.figure(figsize=(8, 7.5))
    axis2 = figure.add_axes([0.1, 0.1, 0.85, 0.85])

    return figure, axis2


def calibrate(fig, ax):

    ax.set_title('Click at the beginning and at the end of scale bar')
    x = fig.ginput(n=2)
    x1 = x[0][0]
    x2 = x[1][0]
    dx = x2 - x1
    ax.set_title('enter the length of the scale bar in micrometers')

    size = input('Type the length of the scale bar in micrometers')

    dx /= size
    return dx


def pixel_size(bar_size, bar_size_pixels):

    """
    computes size of single pixel in the image (in micrometers)
    """

    pix_size = bar_size/bar_size_pixels

    return pix_size


def read_in_settings (fn):

    """
    Read experimental settings from the .txt file, produced by electron microscope
    :param fn: file name
    :returns ...:
    """

    global date, date
    fn += '.txt'
    file = open(fn, 'r')
    for line in file.readlines():
        if re.search('CM_MAG', line):
            magnification = line.split(' ')[-1]
            magnification = np.float(magnification)
            print('Magnification: ', magnification)
        elif re.search('CM_TIME', line):
            zeit = line.split(' ')[-1]
            # zeit = np.float(zeit)
            print('Time of scan: ', zeit)
        elif re.search('CM_DATE', line):
            date = line.split(' ')[-1]
            # date = np.float(date)
            print('Date of scan: ', date)
        elif re.search('SM_MICRON_BAR', line):
            barsize_pixels = line.split(' ')[-1]
            barsize_pixels = np.float(barsize_pixels)
            print('Length of scale bar in pixels: ', barsize_pixels)
        elif re.search('SM_MICRON_MARKER', line):
            barsize = line.split(' ')[-1]
            if re.search('um', barsize):
                barsize = np.float(barsize.split('u')[0])
                print('Length of scale bar in micrometers: ', barsize)

    return magnification, barsize, barsize_pixels, zeit, date


def create_coord_pairs(image_shape):

    """
    converts image size into list of tuples of coordinate pairs
    :param image_shape: shape of np.array containing image
    :return out: list
    """

    a = np.arange(0, image_shape[0])
    b = np.arange(0, image_shape[1])
    out = list(product(a, b))

    return out


def measure_surface(path, im_ix, pix_size):

    patch_points = list(compress(im_ix, path.contains_points(im_ix)))

    area = len(patch_points) * pix_size * pix_size

    return area


def mark_features(fig, ax2, col, path):

    """
    Label features within selected area. Removes those outside the area.
    :param filename:
    :param fig:
    :param ax1:
    :param ax2:
    :param storage:
    :return:
    """

    ax2.set_title('Select features you are interested in. Press ENTER when done')
    x = np.asarray(fig.ginput(n=-1))
    within_patch = list(compress(x, path.contains_points(x)))
    for i in range(len(within_patch)):
        ax2.plot(within_patch[i][0], within_patch[i][1], linestyle='', marker='+', color=col, markersize=10)
    return x


def select_area(filename, fig, ax2, storage):

    """
    Select and measures areas while true
    :param filename:
    :param fig:
    :param ax1:
    :param ax2:
    :param storage:
    :return:
    """
    color_spread1 = np.linspace(0.05, 0.95, 10)
    farba = [cm.Set1(x) for x in color_spread1]
    more = 'y'
    counter = 0
    ph = []
    while more == 'y':
        fig.suptitle(filename)
        ax2.set_title('Select corners of the area you are interested in. Press ENTER when done')
        x = np.asarray(fig.ginput(n=-1))
        ph.append(ax2.add_patch(mpatch.Polygon(x, facecolor=farba[counter], alpha=0.2)))
        ax2.set_title('Are you happy? Press Y to store data or N to drop them')
        happiness = input('Are you happy? Press Y to store data or N to drop them\n')

        if happiness == 'y':
            ax2.set_title('STORING data')
            p = mpath.Path(x)
            surface_area = measure_surface(p, im_index, pix_size)
            storage = store_area(storage, filename, x, surface_area, parallel=counter)
            ax2.set_title('Do you wish to mark features inside area? Y or N')
            mark_yesno = input('Do you wish to mark features inside area? Y or N\n')

            if mark_yesno == 'y':
                features = mark_features(fig, ax2, col=farba[counter], path=p)
                storage = store_features(storage, filename, features, surface_area, parallel=counter)
            del x, p, surface_area

        elif happiness == 'n':
            ax2.set_title('DELETING data')
            ph[counter].remove()
            del x

        ax2.set_title('press Y - select another surface - press N if done', color='blue')
        more = input('press Y - select another surface - press N if done\n')
        if more == 'y':
            counter += 1

    return storage


if __name__ == '__main__':

    #TODO: file selector
    # filename = './samples/Vzorec_120_005.tif'
    filename = './samples/Vzorec_118_009'
    """ read in the image file """
    img = mimg.imread(filename + '.tif')

    """ show loaded image """
    fig, ax2 = drawing_board()
    ax2.imshow(img)
    ax2.set_xlabel('pixels')
    ax2.set_ylabel('pixels')

    """ create img coordinate pairs"""
    im_index = create_coord_pairs(img.shape)

    """ get experimental metadata"""
    magnification, barsize, barsize_pixels, zeit, date = read_in_settings(filename)
    try:
        pix_size = pixel_size(barsize, barsize_pixels)
    except:
        print('There are no calibration data in log file. Starting calibration')
        ax2.set_title('There are no calibration data in log file. Starting calibration')
        pix_size = calibrate(fig, ax2)

    """ prepare empty storage """
    storage = prepare_storage()

    """ action """
    ax2.set_label('Now what? Measure (A)rea, Measure (D)distance, (Q)uit')
    now_what = input('Now what? Measure (A)rea, Measure (D)distance, (Q)uit\n')
    if now_what == 'a':
        storage = select_area(filename, fig, ax2, storage)
    elif now_what == 'd':
        pass
        # TODO: function to measure length
    elif now_what == 'q':
        exit()
        # TODO: check if data in df and save to .csv

