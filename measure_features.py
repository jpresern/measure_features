#!/usr/bin/env python3

import tkinter as tk
import matplotlib

matplotlib.use("TkAgg")

import sys
import re
import pandas as pd
import numpy as np
import math
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.image as mimg
import matplotlib.path as mpath
from itertools import product, compress
from datetime import datetime
from optparse import OptionParser
from tkinter.filedialog import askopenfilename, asksaveasfilename
from datetime import datetime

from IPython import embed

__author__ = "bpiskur, jpresern"


def order_points(pts):
    # list of coordinates 1: top-left,
    # 2: top-right, 3:bottom-right, 4: bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def image_display(image_fn, meta_fn):
    """ read in the image file """
    img = mimg.imread(image_fn)
    # if img.shape[0] > img.shape[1]:
    #     img = PIL.Image.fromarray(img)
    #     img = np.array(img.rotate(90))

    """ show loaded image """
    figure, axis2, axis1 = drawing_board()
    axis2.imshow(img)
    figure.show()

    """ get experimental metadata"""
    mag, bar, bar_pixels, zeitgeist, datum = read_in_settings(meta_fn, fig=figure, ax2=axis2)
    px_size = pixel_size(bar, bar_pixels)

    """ cosmetics """
    axis2.set_xticklabels(np.round(axis2.get_xticks() * px_size[0], 2))
    axis2.set_yticklabels(np.round(axis2.get_yticks() * px_size[1], 2))
    axis2.set_xlabel(r'width [$px$]')
    axis2.set_ylabel(r'height [$px$]')
    axis2.set_title(image_fn)

    """ create img coordinate pairs"""
    im_ind = create_coord_pairs(img.shape)

    return figure, axis2, im_ind, px_size, img


def get_things_saved(figa, store, store_short, file_name, suggested_name):
    """ get filename """
    idir = file_name.rsplit('/', maxsplit=1)[0]
    sname = suggested_name.rsplit('.', maxsplit=1)[0]
    window_save = tk.Tk()
    window_save.withdraw()
    save_fn = asksaveasfilename(initialfile=sname, filetypes=[('All files', '*')], title='Suggest file name',
                                initialdir=idir)
    window_save.destroy()

    store.to_csv(save_fn + '.csv', index_label="index")
    store_short.to_csv(save_fn + '_short.csv', index_label="index")
    """ resets instructions to nothing before save"""
    figa.savefig(save_fn + '.pdf')


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

    columns = ['datetime', 'sample', 'parallel', 'type', 'element', 'x', 'y', 'quality', 'quantity']
    df = pd.DataFrame(columns=columns)

    return df


def prepare_short_storage():
    """
    sets up empty dataframe, containing following columns:
        sample      - name of photo
        parallel    - iteration on the same sample
        quality     - quality measured (surface if area, density of points inside the area if points, distance if distance)
        quantity    - measured value for quality
    :return df: pandas dataframe, ready
    """

    columns = ['datetime', 'sample', 'parallel', 'quality', 'quantity']
    df = pd.DataFrame(columns=columns)

    return df


def store_area(df, df2, file, xy_pairs, area, parallel=1):
    """
    Appends measurements to df (full data storage) and df2 (short data storage)
    :param df:
    :param df2:
    :param file:
    :param xy_pairs:
    :param area:
    :param parallel:
    :return:
    """
    columns = ['datetime', 'sample', 'parallel', 'type', 'element', 'x', 'y', 'quality', 'quantity']
    columns2 = ['datetime', 'sample', 'parallel', 'quality', 'quantity']
    df_empty = pd.DataFrame(columns=columns)
    df_empty2 = pd.DataFrame(columns=columns2)
    for i, v in enumerate(xy_pairs):
        df_empty.loc[i, :] = datetime.now(), file, parallel, 'area', 'corner', v[0], v[1], 'surface', area
    df_empty2.loc[0, :] = datetime.now(), file, parallel, 'surface', area

    return pd.concat([df, df_empty], ignore_index=True), pd.concat([df2, df_empty2], ignore_index=True)


def store_features(df, df2, file, xy_pairs, area, parallel=1):
    """
    appends feature count and density fo full storage (df) and short storage (df2)
    :param df:
    :param df2:
    :param file:
    :param xy_pairs:
    :param area:
    :param parallel:
    :return:
    """
    columns = ['datetime', 'sample', 'parallel', 'type', 'element', 'x', 'y', 'quality', 'quantity']
    columns2 = ['datetime', 'sample', 'parallel', 'quality', 'quantity']
    df_empty = pd.DataFrame(columns=columns)
    df_empty2 = pd.DataFrame(columns=columns2)
    density = xy_pairs.shape[0] / area
    for i, v in enumerate(xy_pairs):
        df_empty.loc[i, :] = datetime.now(), file, parallel, 'points', 'points', v[0], v[1], 'density', density
    df_empty2.loc[0, :] = datetime.now(), file, parallel, 'density', density
    df_empty2.loc[1, :] = datetime.now(), file, parallel, 'count', xy_pairs.shape[0]

    return pd.concat([df, df_empty], ignore_index=True), pd.concat([df2, df_empty2], ignore_index=True)


def store_distance(df, df2, file, xy_pairs, distance, parallel=1):
    """
    :param df:
    :param df2:
    :param file:
    :param xy_pairs:
    :param distance:
    :param parallel:
    :return:
    """
    columns = ['datetime', 'sample', 'parallel', 'type', 'element', 'x', 'y', 'quality', 'quantity']
    columns2 = ['datetime', 'sample', 'parallel', 'quality', 'quantity']
    df_empty = pd.DataFrame(columns=columns)
    df_empty2 = pd.DataFrame(columns=columns2)
    df_empty.loc[0, :] = datetime.now(), file, parallel, 'distance', 'start_point', xy_pairs[0, 0], xy_pairs[0, 1], \
                         'length', distance
    df_empty.loc[1, :] = datetime.now(), file, parallel, 'distance', 'end_point', xy_pairs[1, 0], xy_pairs[1, 1], \
                         'length', distance
    df_empty2.loc[0, :] = datetime.now(), file, parallel, 'length', distance

    return pd.concat([df, df_empty], ignore_index=True), pd.concat([df2, df_empty2], ignore_index=True)


def store_ci(df, df2, file, xy_pairs, ci, parallel=1):

    columns = ['datetime', 'sample', 'parallel', 'type', 'element', 'x', 'y', 'quality', 'quantity']
    columns2 = ['datetime', 'sample', 'parallel', 'quality', 'quantity']
    df_empty = pd.DataFrame(columns=columns)
    df_empty2 = pd.DataFrame(columns=columns2)
    df_empty.loc[0, :] = datetime.now(), file, parallel, 'ci', 'start_point', xy_pairs[0, 0], xy_pairs[0, 1], \
                         'ci', ci
    df_empty.loc[1, :] = datetime.now(), file, parallel, 'ci', 'common_point', xy_pairs[1, 0], xy_pairs[1, 1], \
                         'ci', ci
    df_empty.loc[2, :] = datetime.now(), file, parallel, 'ci', 'end_point', xy_pairs[2, 0], xy_pairs[2, 1], \
                         'ci', ci
    df_empty2.loc[0, :] = datetime.now(), file, parallel, 'ci', ci

    return pd.concat([df, df_empty], ignore_index=True), pd.concat([df2, df_empty2], ignore_index=True)


def drawing_board():
    """ create image with axes """

    # TODO: separate axis for image and for labeling. This will allow layering of .pdf, hopefully

    figure = plt.figure(figsize=(8, 8))
    axis2 = figure.add_axes([0.1, 0.1, 0.85, 0.85])
    # axis1 = axis2.twinx()
    # axis1.set_yticks([])
    axis1 = axis2

    return figure, axis2, axis1


def calibrate(figa, ax):
    figa.canvas.manager.window.deiconify()
    figa.canvas.manager.window.tkraise()
    ax.set_title('Click at the beginning and at the end of scale bar')
    x = figa.ginput(n=2)
    x1 = x[0][0]
    x2 = x[1][0]
    dx = x2 - x1
    ax.set_title('enter the length of the scale bar in micrometers')

    size = input('Type the length of the scale bar in micrometers')

    pixel_size = dx / np.float(size)
    return dx, size, pixel_size


def pixel_size(bar_size, bar_size_pixels):
    """
    computes size of single pixel in the image (in micrometers)
    """

    pix_size = bar_size / bar_size_pixels

    return (pix_size, pix_size)


def read_in_settings(file_name, fig, ax2):
    """
    Read experimental settings from the .txt file, produced by electron microscope
    :param fn: file name
    :returns ...:
    """
    barsize = None
    barsize_pixels = None

    # global date, date
    file_name += '.txt'
    try:
        try:
            file = open(file_name, mode='r', encoding="utf8")
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
            file = open(file_name, mode='r', encoding="cp1252")
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
        print('There are no calibration data. Everything will be set at value 1')
        magnification = 1
        barsize_pixels = 1
        barsize = 1
        zeit = None
        date = None



    # if (barsize == None) | (barsize_pixels == None):
    #     print('There are no calibration data in log file.')
    #     want_calib = input('Calibrate? Y/N\n')
    #     if want_calib == 'y':
    #         get_things_saved(fig, storage, storage_short, filename, title_fn)
    #         ax2.set_title('There are no calibration data in log file. Starting calibration')
    #         barsize, barsize_pixels, pix_size = calibrate(fig, ax2)
    #     elif want_calib == 'n':
    #         barsize, barsize_pixels, pix_size, magnification = 1, 1, 1, 1
    #         zeit = None
    #         date = None
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


def measure_surface(path, im_ix, pixie_size):

    patch_points = list(compress(im_ix, path.contains_points(im_ix)))
    area = len(patch_points) * pixie_size[0] * pixie_size[1]

    return area


def mark_features(figa, axis2, colore, drawn_path):
    """
    Label features within selected area. Removes those outside the area.
    :param filename:
    :param figa:
    :param axis1:
    :param axis2:
    :param colore:
    :return:
    """
    figa.canvas.manager.window.deiconify()
    figa.canvas.manager.window.tkraise()
    axis2.set_title('Select features you are interested in. Press ENTER when done')
    x = np.asarray(fig.ginput(n=-1, timeout=0))
    within_patch = list(compress(x, drawn_path.contains_points(x)))
    for i in range(len(within_patch)):
        axis2.plot(within_patch[i][0], within_patch[i][1], linestyle='', marker='+', color=colore, markersize=8)
    return x


def select_area(file, figa, axa2, store, store_short, im_ix, pixsize=1, count=0):
    """
    Select and measures areas while true
    :param file:
    :param figa:
    :param axa2:
    :param store:
    :param store_short:
    :param im_ix:
    :param pixsize:
    :param count:
    :return:
    """
    color_spread1 = np.linspace(0.05, 0.95, 10)
    farba = [cm.Set1(x) for x in color_spread1]
    ph = []
    ph_count = 0
    # figa.canvas.manager.window.deiconify()
    # figa.canvas.manager.window.tkraise()
    figa.suptitle(file)
    axa2.set_title('Select corners of the area you are interested in. Press ENTER when done')
    x = np.asarray(figa.ginput(n=-1, timeout=0))

    """ conversion to polar coordinates and sort, convert back"""
    x = list(tuple(map(tuple, x)))
    cent = (sum([xx[0] for xx in x]) / len(x), sum([xx[1] for xx in x]) / len(x))
    # sort by polar angle
    x.sort(key=lambda xx: math.atan2(xx[1] - cent[1], xx[0] - cent[0]))
    x = np.array(x)

    ph.append(axa2.add_patch(mpatch.Polygon(x, facecolor=farba[count % len(farba)], alpha=0.2)))
    # figa.canvas.manager.window.iconify()
    axa2.set_title('Are you happy? Press Y to store data or N to drop them')
    happiness = input('Are you happy? Press Y to store data or N to drop them\n')

    if happiness == 'y':
        axa2.set_title('STORING data')
        p = mpath.Path(x)
        surface_area = measure_surface(p, im_ix, pixsize)
        print('Area size [um2]: ', surface_area)
        store, store_short = store_area(store, store_short, file, x, surface_area, parallel=count)
        axa2.set_title('Do you wish to mark features inside area? Y or N')
        mark_yesno = input('Do you wish to mark features inside area? Y or N\n')

        if mark_yesno == 'y':
            # figa.canvas.manager.window.deiconify()
            # figa.canvas.manager.window.tkraise()
            features = mark_features(figa, axa2, colore=farba[count % len(farba)], drawn_path=p)
            store, store_short = store_features(store, store_short, file, features, surface_area, parallel=count)
            # figa.canvas.manager.window.iconify()
        del x, p, surface_area

    elif happiness == 'n':
        axa2.set_title('DELETING data')
        ph[ph_count].remove()
        del x

    return store, store_short, count, figa, axa2


def measure_distance(file, figa, axis2, store, store_short, pix=1, count=0):
    axis2.set_title('Measure distance between two points')
    more = 'y'
    while more == 'y':
        # figa.canvas.manager.window.deiconify()
        # figa.canvas.manager.window.tkraise()
        xy = np.asarray(figa.ginput(n=2, timeout=0))
        # dist = np.linalg.norm(xy[0] - xy[1])
        # dist_test = np.linalg.norm(xy[0]*pix[0] - xy[1]*pix[1])

        dist = np.linalg.norm(xy[0]*pix[0] - xy[1]*pix[1])

        dist *= pix[0]
        linija = axis2.plot(xy[:, 0], xy[:, 1], marker="+", markersize=8)
        axis2.text(np.mean(xy, axis=0)[0], np.mean(xy, axis=0)[1], str(np.round(dist, 1)) + r' $\mu$m',
                   color=linija[0].get_color())
        # axis2.text(np.mean(xy, axis=0)[0] + 0.05, np.mean(xy, axis=0)[1] +0.05, str(np.round(dist_test, 1)) + r' $\mu$m',
        #            color=linija[0].get_color())

        store, store_short = store_distance(store, store_short, file, xy, distance=dist, parallel=count)
        count += 1
        # figa.canvas.manager.window.iconify()
        axis2.set_title('One more? Y/N')
        more = input('One more? Y/N\n')
    axis2.set_title('')
    return store, store_short, count, figa, axis2


def measure_ci(file, figa, axis2, store, store_short, pix=1, count=0):
    axis2.set_title('Select three points of CI')

    figa.canvas.manager.window.deiconify()
    figa.canvas.manager.window.tkraise()
    xy = np.asarray(figa.ginput(n=3, timeout=0))
    dist0 = np.linalg.norm(xy[0] - xy[1])
    dist0 *= pix
    dist1 = np.linalg.norm(xy[1] - xy[2])
    dist1 *= pix
    ci = dist0/dist1
    print('CI: ' + str(np.round(ci,2)) +'\n')
    linija0 = axis2.plot(xy[:, 0], xy[:, 1], marker="+", markersize=8, color='g')
    # linija1 = axis2.plot(xy[:,1], xy[:, 2], marker="+", markersize=8, color='r')

    axis2.text(np.mean(xy, axis=0)[0], np.mean(xy, axis=0)[1], str(np.round(ci, 2)),
               color=linija0[0].get_color())
    figa.canvas.manager.window.iconify()
    axis2.set_title('Are you happy? Press Y to store data or N to drop them')
    happiness = input('Are you happy? Press Y to store data or N to drop them\n')

    if happiness == 'y':
        axis2.set_title('STORING data')
        store, store_short = store_ci(store, store_short, file, xy, ci=ci, parallel=count)
        count += 1

    elif happiness == 'n':
        del ci, dist0, dist1
        linija0[0].remove()

    axis2.set_title('')
    return store, store_short, count


def bulk_measure_ci(file, figa, axis2, store, store_short, pix=1, count=0):
    happiness = 'n'
    while happiness == 'n':
        axis2.set_title(file + ': Select three points of CI')
        figa.canvas.manager.window.deiconify()
        figa.canvas.manager.window.tkraise()
        xy = np.asarray(figa.ginput(n=3, timeout=0))
        dist0 = np.linalg.norm(xy[0] - xy[1])
        dist0 *= pix
        dist1 = np.linalg.norm(xy[1] - xy[2])
        dist1 *= pix
        ci = dist0/dist1
        print('CI: ' + str(np.round(ci,2)) +'\n')
        linija0 = axis2.plot(xy[:, 0], xy[:, 1], marker="+", markersize=8, color='g')
        # linija1 = axis2.plot(xy[:,1], xy[:, 2], marker="+", markersize=8, color='r')

        tekst0 = axis2.text(np.mean(xy, axis=0)[0], np.mean(xy, axis=0)[1], str(np.round(ci, 2)),
                   color=linija0[0].get_color())
        figa.canvas.manager.window.iconify()
        axis2.set_title('Are you happy? Press Y to store data or N to drop them')
        happiness = input('Are you happy? Press Y to store data or N to drop them\n')

        if happiness == 'n':
            del ci, dist0, dist1
            linija0[0].remove()
            tekst0[0].remove()

        if happiness == 'y':
            axis2.set_title('STORING data')
            store, store_short = store_ci(store, store_short, file, xy, ci=ci, parallel=count)
            count += 1

    axis2.set_title('')
    return store, store_short, count


def redraw_stored_things(axis2, store):

    """ draw selecteda areas with marked features
        sample      - name of photo
        parallel    - iteration on the same sample
        type        - what are we measuring (area, points, distance)
        x, y        - coordinates
        quality     - quality measured (surface if area, density of points inside the area if points, distance if distance)
        quantity    - measured value for quality
    """
    color_spread1 = np.linspace(0.05, 0.95, 10)
    farba = [cm.Set1(x) for x in color_spread1]
    count = 0


    elements = store["parallel"].unique()

    for element in elements:

        iter_element = store.loc[store["parallel"] == element]
        element_type = iter_element["type"].unique().tolist()

        if element_type.__contains__("area"):

            if element_type.__contains__("points"):
                points = iter_element.loc[(iter_element["type"] == "points"), ["x", "y"]].values
                axis2.plot(points[:, 0], points[:, 1], linestyle="", marker="+", markersize=8, color=farba[count])

        elif element_type.__contains__("distance"):
            points = iter_element.loc[(iter_element["type"] == "distance"), ["x", "y"]].values
            dist = np.linalg.norm(points[0,:] - points[1,:])
            axis2.plot(points[:, 0], points[:, 1], marker="+", markersize=8, color=farba[count])
            axis2.text(np.mean(points, axis=0)[0], np.mean(points, axis=0)[1], str(np.round(dist, 1)) + r' $\mu$m',
                       color=farba[count])
        count +=1
    pass


def load_measurements(file, file2):

    """ load data """

    store = pd.read_csv(file, usecols=['datetime', 'sample', 'parallel', 'type', 'element', 'x', 'y', 'quality',
                                         'quantity'])
    store_short = pd.read_csv(file2, usecols=['datetime', 'sample', 'parallel', 'quality', 'quantity'])

    """ construct file names """
    impath = file.rsplit('/', maxsplit=1)[0]
    sample_fn = store["sample"].values[0]
    im_fn = impath + "/" + sample_fn
    meta_fn = impath + "/" + sample_fn.rsplit(".")[0]
    """ draw """
    figa, axis, im_ix, pix = image_display(im_fn, meta_fn)
    """ draw measured features """
    redraw_stored_things(axis, store)

    return store, store_short, figa, axis, pix, im_ix


if __name__ == '__main__':

    """ declare myself """

    print("=============================================================")
    print("|    Measure features - Simple tool for simple jobs v.1.1   |")
    print("|                                                           |")
    print("|    by: Barbara Piskur and Janez Presern (c) 2016, 2017    |")
    print("|                                                           |")
    print("|    read README.md for help and instructions               |")
    print("=============================================================")
    args = sys.argv
    to_be_parsed = args[1:]

    """ define options parser """
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

        """ load existing .csv file """
        if ext == '.csv':

            """ construct file names for storage and storage_short """
            if "_short" not in fn:
                filename1 = filename
                filename2 = fn + '_short' + ext
            else:
                filename1 = filename.rsplit('_short.', maxsplit=1)[0] + filename.rsplit('_short', maxsplit=1)[1]
                filename2 = filename

            storage, storage_short, fig, ax2, pix_size, im_index = load_measurements(filename1, filename2)
            counter = storage["parallel"].max() + 1
        else:

            """ load image, load meta data, display image and calibrate """
            fig, ax2, im_index, pix_size, imago = image_display(filename, fn)

            """ prepare empty storage """
            storage = prepare_storage()
            storage_short = prepare_short_storage()
            counter = 0

        """ action """
        """ main loop """
        stay = True
        while stay:
            fig.canvas.manager.window.iconify()
            ax2.set_title('Now what? Measure (A)rea, count (B)ees, cubital (I)ndex, Measure (L)ength, (C)lose image')
            now_what = input('Now what? Measure (A)rea, count (B)ees, cubital (I)ndex, Measure (L)ength, (C)lose image\n')
            if now_what == 'a':
                fig.canvas.manager.window.deiconify()
                fig.canvas.manager.window.tkraise()
                storage, storage_short, counter = select_area(title_fn, fig, ax2, storage, storage_short, im_index,
                                                              pix_size, counter)

            elif now_what == 'b':
                fig.canvas.manager.window.deiconify()
                fig.canvas.manager.window.tkraise()
                stretched = stretch_comb(filename, fig, ax2)
                storage, storage_short, counter = select_area(title_fn, fig, ax2, storage, storage_short, im_index,
                                                              pix_size, counter)
            elif now_what == 'i':
                fig.canvas.manager.window.deiconify()
                fig.canvas.manager.window.tkraise()
                storage, storage_short, counter = measure_ci(title_fn, fig, ax2, storage, storage_short, pix_size,
                                                             counter)

            elif now_what == 'l':
                fig.canvas.manager.window.deiconify()
                fig.canvas.manager.window.tkraise()
                storage, storage_short, counter = measure_distance(title_fn, fig, ax2, storage, storage_short, pix_size,
                                                                   counter)
            elif now_what == 'c':
                ax2.set_title('Save measurements? Y/N')
                safe = input('Save measurements? Y/N\n')
                if safe == 'y':
                    ax2.set_title('')
                    get_things_saved(fig, storage, storage_short, filename, title_fn)
                elif safe == 'n':
                    pass
                stay = False
                plt.close(fig)
            counter += 1
        und_jetzt = input('And now what? (L)oad new image, or (Q)uit?\n')
        if (und_jetzt == 'q') or (und_jetzt == 'Q'):
            do_work = False

    exit()
