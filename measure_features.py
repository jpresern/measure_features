#!/usr/bin/env python3


import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")

import sys
import re
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.image as mimg
import matplotlib.path as mpath
from itertools import product, compress
from datetime import datetime
from optparse import OptionParser
from tkinter.filedialog import askopenfilename, asksaveasfilename
from IPython import embed

__author__ = "jpresern, bpiskur"


def image_display(image_fn, meta_fn):

    """ read in the image file """
    img = mimg.imread(image_fn)

    """ show loaded image """
    figure, axis2, axis1 = drawing_board()
    axis1.imshow(img)
    figure.show()

    """ get experimental metadata"""
    mag, bar, bar_pixels, zeitgeist, datum = read_in_settings(meta_fn)
    pix_size = pixel_size(bar, bar_pixels)

    axis2.set_xticklabels(np.round(axis2.get_xticks() * pix_size, 2))
    axis2.set_yticklabels(np.round(axis2.get_yticks() * pix_size, 2))
    axis2.set_xlabel(r'width [$\mu$m]')
    axis2.set_ylabel(r'height [$\mu$m]')

    """ create img coordinate pairs"""
    im_ind = create_coord_pairs(img.shape)

    return figure, axis2, im_ind, pix_size


def get_things_saved(fig, storage, file_name, suggested_name):
    """ get filename """
    idir = file_name.rsplit('/', maxsplit=1)[0]
    sname = suggested_name.rsplit('.', maxsplit=1)[0]
    window_save = tk.Tk()
    window_save.withdraw()
    save_fn = asksaveasfilename(initialfile=sname, filetypes=[('All files', '*')], title='Suggest file name',
                                initialdir=idir)
    window_save.destroy()

    storage.to_csv(save_fn + '.csv')
    """ resets instructions to nothing before save"""
    fig.savefig(save_fn + '.pdf')


def get_things_opened():

    idir = './samples'
    window_open = tk.Tk()
    window_open.withdraw()
    im_name = askopenfilename(initialdir=idir, title='Select image file')
    window_open.destroy()

    return im_name


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

    columns = ['datetime','sample', 'parallel', 'type', 'element', 'x', 'y', 'quality', 'quantity']
    df = pd.DataFrame(columns=columns)

    return df


def store_area(df, fn, xy_pairs, area, parallel=1):
    columns = ['datetime','sample', 'parallel', 'type', 'element', 'x', 'y', 'quality', 'quantity']
    df_empty = pd.DataFrame(columns=columns)
    for i, v in enumerate(xy_pairs):
        df_empty.loc[i, :] = datetime.now(), fn, parallel, 'area', 'corner', v[0], v[1], 'surface', area
    return pd.concat([df, df_empty], ignore_index=True)


def store_features(df, fn, xy_pairs, area, parallel=1):
    columns = ['datetime','sample', 'parallel', 'type', 'element', 'x', 'y', 'quality', 'quantity']
    df_empty = pd.DataFrame(columns=columns)
    density = xy_pairs.shape[0]/area
    for i, v in enumerate(xy_pairs):
        df_empty.loc[i, :] = datetime.now(), fn, parallel, 'points', 'points', v[0], v[1], 'density', density
    return pd.concat([df, df_empty], ignore_index=True)


def store_distance(df, fn, xy_pairs, distance, parallel=1):
    columns = ['datetime','sample', 'parallel', 'type', 'element', 'x', 'y', 'quality', 'quantity']
    df_empty = pd.DataFrame(columns=columns)
    df_empty.loc[0, :] = datetime.now(), fn, parallel, 'distance', 'start_point', xy_pairs[0, 0], xy_pairs[0, 1], \
                         'length', distance
    df_empty.loc[1, :] = datetime.now(), fn, parallel, 'distance', 'end_point', xy_pairs[1, 0], xy_pairs[1, 1], \
                         'length', distance
    return pd.concat([df, df_empty], ignore_index=True)


def drawing_board():
    """ create image with axes """

    #TODO: separate axis for image and for labeling. This will allow layering of .pdf, hopefully

    figure = plt.figure(figsize=(8, 7.5))
    axis2 = figure.add_axes([0.1, 0.1, 0.85, 0.85])
    # axis1 = axis2.twinx()
    # axis1.set_yticks([])
    axis1 = axis2

    return figure, axis2, axis1


def calibrate(fig, ax):

    ax.set_title('Click at the beginning and at the end of scale bar')
    x = fig.ginput(n=2)
    x1 = x[0][0]
    x2 = x[1][0]
    dx = x2 - x1
    ax.set_title('enter the length of the scale bar in micrometers')

    size = input('Type the length of the scale bar in micrometers')

    pixel_size = dx/np.float(size)
    return dx, size, pixel_size


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
    barsize = None
    barsize_pixels = None

    # global date, date
    fn = fn + '.txt'
    try:
        file = open(fn, 'r')

        for line in file.readlines():
            if re.search('CM_MAG', line):
                magnification = line.split(' ')[-1]
                magnification = np.float(magnification)
                print('Magnification: ', magnification)
            else:
                magnification = 'data not available'
            if re.search('CM_TIME', line):
                zeit = line.split(' ')[-1]
                # zeit = np.float(zeit)
                print('Time of scan: ', zeit)
            else:
                zeit = 'data not available'
            if re.search('CM_DATE', line):
                date = line.split(' ')[-1]
                # date = np.float(date)
                print('Date of scan: ', date)
            else:
                date = 'data not available'
            if re.search('SM_MICRON_BAR', line):
                barsize_pixels = line.split(' ')[-1]
                barsize_pixels = np.float(barsize_pixels)
                print('Length of scale bar in pixels: ', barsize_pixels)
            if re.search('SM_MICRON_MARKER', line):
                barsize = line.split(' ')[-1]
                if re.search('um', barsize):
                    barsize = np.float(barsize.split('u')[0])
                    print('Length of scale bar in micrometers:\n', barsize)
    except:
        pass

    if (barsize == None) | (barsize_pixels == None):

        print('There are no calibration data in log file. Starting calibration')
        ax2.set_title('There are no calibration data in log file. Starting calibration')
        barsize, barsize_pixels, pix_size = calibrate(fig, ax2)
        magnification = 1

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
    fig.canvas.manager.window.tkraise()
    ax2.set_title('Select features you are interested in. Press ENTER when done')
    x = np.asarray(fig.ginput(n=-1, timeout=0))
    within_patch = list(compress(x, path.contains_points(x)))
    for i in range(len(within_patch)):
        ax2.plot(within_patch[i][0], within_patch[i][1], linestyle='', marker='+', color=col, markersize=8)
    return x


def select_area(file, figa, axa2, store, im_ix, pixsize=1, counter=0):

    """
    Select and measures areas while true
    :param file:
    :param figa:
    :param axa2:
    :param store:
    :param im_ix:
    :param pixsize:
    :param counter:
    :return:
    """
    color_spread1 = np.linspace(0.05, 0.95, 10)
    farba = [cm.Set1(x) for x in color_spread1]
    ph = []

    # embed()
    # fig.canvas.manager.window.focus()
    figa.canvas.manager.window.tkraise()
    figa.suptitle(file)
    axa2.set_title('Select corners of the area you are interested in. Press ENTER when done')
    x = np.asarray(fig.ginput(n=-1, timeout=0))
    ph.append(axa2.add_patch(mpatch.Polygon(x, facecolor=farba[counter], alpha=0.2)))
    axa2.set_title('Are you happy? Press Y to store data or N to drop them')
    happiness = input('Are you happy? Press Y to store data or N to drop them\n')

    if happiness == 'y':
        axa2.set_title('STORING data')
        p = mpath.Path(x)
        surface_area = measure_surface(p, im_ix, pixsize)
        store = store_area(store, file, x, surface_area, parallel=counter)
        axa2.set_title('Do you wish to mark features inside area? Y or N')
        mark_yesno = input('Do you wish to mark features inside area? Y or N\n')

        if mark_yesno == 'y':
            features = mark_features(figa, axa2, col=farba[counter], path=p)
            store = store_features(store, file, features, surface_area, parallel=counter)
        del x, p, surface_area

    elif happiness == 'n':
        axa2.set_title('DELETING data')
        ph[counter].remove()
        del x

    return store, counter


def measure_distance(filename, fig, ax2, storage, pix_size=1, counter=0):
    # todo: add micrometer sign
    ax2.set_title('Measure distance between two points')
    more = 'y'
    while more == 'y':
        fig.canvas.manager.window.tkraise()
        xy = np.asarray(fig.ginput(n=2, timeout=0))
        dist = np.linalg.norm(xy[0] - xy[1])
        dist *= pix_size
        linija = ax2.plot(xy[:, 0], xy[:, 1], marker="+", markersize=8)
        ax2.text(np.mean(xy, axis = 0)[0], np.mean(xy, axis = 0)[1], str(np.round(dist, 0)) + r' $\mu$m',
                 color=linija[0].get_color())
        storage = store_distance(storage, filename, xy, distance=dist, parallel=counter)
        counter += 1
        ax2.set_title('One more? Y/N')
        more = input('One more? Y/N\n')
    ax2.set_title('')
    return storage, counter


if __name__ == '__main__':

    args = sys.argv
    to_be_parsed = args[1:]

    # define options parser
    parser = OptionParser()
    parser.add_option("-f", "--file", action="store", type="string", dest="filename",
                      default='')

    (options, args) = parser.parse_args(args)

    """ outer main loop """
    do_work = True
    while do_work:

        if options.filename:
            filename = options.filename

        else:
            filename = get_things_opened()


        """ sort extensions etc """
        fn = filename.rsplit('.', maxsplit=1)[0]
        ext = filename.rsplit('.', maxsplit=1)[1]
        ext = '.' + ext
        title_fn = filename.rsplit('/', maxsplit=1)[-1]

        """ load image, load meta data, display image and calibrate """
        fig, ax2, im_index, pix_size = image_display(filename, fn)

        """ prepare empty storage """
        storage = prepare_storage()

        """ action """
        """ main loop """
        stay = True
        counter = 0
        while stay:
            ax2.set_title('Now what? Measure (A)rea, Measure (L)ength, (C)lose image')
            now_what = input('Now what? Measure (A)rea, Measure (L)ength, (C)lose image\n')
            if now_what == 'a':
                storage, counter = select_area(title_fn, fig, ax2, storage, im_index, pix_size, counter)
            elif now_what == 'l':
                storage, counter = measure_distance(title_fn, fig, ax2, storage, pix_size, counter)
            elif now_what == 'c':
                ax2.set_title('Save measurements? Y/N')
                safe = input('Save measurements? Y/N\n')
                if safe == 'y':
                    ax2.set_title('')
                    get_things_saved(fig, storage, filename, title_fn)
                elif safe == 'n':
                    pass
                stay = False
                plt.close(fig)
            counter += 1
        und_jetzt = input('And now what? (L)oad new image, or (Q)uit?\n')
        if (und_jetzt == 'q') or (und_jetzt == 'Q'):
            do_work = False

    exit()




