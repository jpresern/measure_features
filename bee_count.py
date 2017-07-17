#!/usr/bin/env python3

import tkinter as tk
import matplotlib

matplotlib.use("TkAgg")

import sys, os, glob
import re
import pandas as pd
import numpy as np
import math
import cv2
import PIL
import imutils
import PIL.ImageChops as ic
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.image as mimg
import matplotlib.path as mpath
from itertools import product, compress
from datetime import datetime
from optparse import OptionParser
from tkinter.filedialog import askopenfilename, asksaveasfilename, askdirectory
from datetime import datetime
from IPython import embed
from measure_features import get_things_opened, bulk_measure_ci, image_display, prepare_storage, \
    prepare_short_storage, four_point_transform, create_coord_pairs, drawing_board, select_area, \
    measure_distance, store_area, store_distance, store_features, get_things_saved

__author__ = "bpiskur, jpresern"


def superimpose_mask_on_image(mask, image, color_delta=[-20, 40, -20], invert=False):
    # superimpose mask on image, the color change being controlled by color_delta
    # TODO: currently only works on 3-channel, 8 bit images and 1-channel, 8 bit masks
    mask = mask.astype('float')

    if invert == 'False':
        mask[mask == 0] = 1
        mask[mask == 255] = 0
    elif invert == 'True':
        mask[mask == 0] = 0
        mask[mask == 255] = 1

    # mask = np.array(ic.invert(PIL.Image.fromarray(mask)))

    # im = np.zeros((image.shape[0], image.shape[1], 3))
    im = np.copy(image)

    # fast, but can't handle overflows
    # im[:,:,0] = np.clip(image[:,:,0] + color_delta[0] * (mask[:,:]), 0, 255)
    # im[:,:,1] = np.clip(image[:,:,1] + color_delta[1] * (mask[:,:]), 0, 255)
    # im[:,:,2] = np.clip(image[:,:,2] + color_delta[2] * (mask[:,:]), 0, 255)

    im[:,:,0] = color_delta[0] * (mask[:,:])
    im[:,:,1] = color_delta[1] * (mask[:,:])
    im[:,:,2] = color_delta[2] * (mask[:,:])

    return image - im


def superimpose_mask(mask, image, color_delta=[40, -20, -20]):
    # superimpose mask on image, the color change being controlled by color_delta
    # TODO: currently only works on 3-channel, 8 bit images and 1-channel, 8 bit masks
    mask = mask.astype('float')
    mask[mask == 0] = 1
    mask[mask == 255] = 0

    # mask = np.array(ic.invert(PIL.Image.fromarray(mask)))

    # im = np.zeros((image.shape[0], image.shape[1], 3))
    im = np.copy(image)

    # fast, but can't handle overflows
    im[:,:,0] = np.clip(image[:,:,0] + color_delta[0] * (mask[:,:]), 0, 255)

    return im


def multichannel_fig ():
    f, ((ax_r, ax_g), (ax_b, ax_gr)) = plt.subplots(2, 2, sharex='col', sharey='row')
    # f.subplots_adjust(hspace=0)
    # f.subplots_adjust(vspace=0)
    fig.set_size_inches(10.5, 10.5, forward=True)
    # f.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    return f, ax_r, ax_g, ax_b, ax_gr


def select_comb_corners(file, figa, axa2):
    """
    Select internal comb corners

    """
    figa.suptitle(file)
    axa2.set_title('Select internal corners of the comb in succession TL, TR, BR, BL')
    happiness = 'n'
    while happiness == 'n':
        # figa.canvas.manager.window.deiconify()
        # figa.canvas.manager.window.tkraise()
        x = np.asarray(fig.ginput(n=-4, timeout=0))
        # figa.canvas.manager.window.iconify()
        axa2.set_title('Are you happy? Press Y to store data or N to drop them')
        happiness = input('Are you happy? Press Y to store data or N to drop them\n')

        if happiness == 'n':
            del x

    axa2.set_title('')
    return x


def stretch_comb(fname, fg, axa):
    """ stretch picture and calibrate picture"""

    corners = select_comb_corners(fname, fg, axa)
    new_img = four_point_transform(mimg.imread(fname), corners)
    im_ix = create_coord_pairs(new_img.shape)
    # fg.canvas.manager.window.iconify()
    frame_dim = input('Select frame dimensions: (A)Z, (L)R full, (D)B\n')
    axa.clear()
    axa.imshow(new_img)

    if frame_dim == 'a':
        # inner dimensions: 244 x 380
        px_size = 0.244 / new_img.shape[0], 0.380 / new_img.shape[1]
        xtcks = axa.get_xticks()
        ytcks = axa.get_yticks()

        axa.set_xticklabels(np.round(xtcks * px_size[0], 2))
        axa.set_yticklabels(np.round(ytcks * px_size[1], 2))

        axa.set_xlabel(r'width $[m]$')
        axa.set_ylabel(r'height $[m]$')

        pass

    elif frame_dim == 'l':
        # todo: add other inner dimensions
        px_size = (1, 1)
        pass

    elif frame_dim == 'd':
        px_size = (1, 1)
        pass

    else:
        px_size = (1, 1)

    # plt.show(fg)
    axa.set_title('Stretched: ' + fname)
    # fig.canvas.manager.window.tkraise()
    # fig.canvas.manager.window.deiconify()
    fig.canvas.draw()
    # fig.canvas.manager.window.iconify()
    return new_img, im_ix, frame_dim, px_size, fg, axa


def mark_bees(fig, ax):
    """
    Label features within selected area. Removes those outside the area.
    :param fig:
    :return:
    """
    x = np.asarray(fig.ginput(n=-1, timeout=0))
    return x


def get_bee_coverage(file, img, fg, ax, px_size, store, store_short):

    count = 0

    #   clear text
    ax.set_title('')

    #   create multichannel display
    # fig2, red, green, blue, gray = multichannel_fig()

    # background equalization
    max_value = np.max(img)
    backgroundRemoved = img.astype(float)
    blur = cv2.GaussianBlur(backgroundRemoved, (151, 151), 50)
    backgroundRemoved = backgroundRemoved / blur
    img_new = (backgroundRemoved * max_value / np.max(backgroundRemoved)).astype(np.uint8)
    #
    #   split and convert image
    im_red = img_new[:,:,0]
    im_green = img_new[:, :, 1]
    im_blue = img_new[:, :, 2]
    im_gray = cv2.cvtColor(img_new, cv2.COLOR_RGB2GRAY)

    #   display images
    # red.imshow(im_red, cmap='gray')
    # red.text(0.5, 0.9, 'Red channel', transform=red.transAxes, color='r')
    # green.imshow(im_green, cmap='gray')
    # green.text(0.5, 0.9, 'Green channel', transform=green.transAxes, color='g')
    # blue.imshow(im_blue, cmap='gray')
    # blue.text(0.5, 0.9, 'Blue channel', transform=blue.transAxes, color='b')
    # gray.imshow(im_gray, cmap='gray')

    del blur, backgroundRemoved


    happy = 'n'
    while happy == 'n':
        ax.clear()
        ax.imshow(img_new)
        fg.canvas.draw()
        block = input('Enter threshold value (0-255)\n')
        block2 = input('Invert selection Y/N\n')


        #TODO: allow input of various parameters included in threshold r22, gr10, etc
        #   thresholding
        _, thresh_red = cv2.threshold(im_red, np.int(block), 255, cv2.THRESH_BINARY)
        _, thresh_gray = cv2.threshold(im_gray, np.int(block), 255, cv2.THRESH_BINARY)

        #   MATH
        out = ic.multiply(PIL.Image.fromarray(thresh_red), PIL.Image.fromarray(thresh_gray))
        out2 = ic.subtract(PIL.Image.fromarray(thresh_red), PIL.Image.fromarray(thresh_gray))
        out_light = ic.lighter(out, out2)

        # #   erosion + dilation
        kernel = np.ones((5, 5), np.uint8)
        out_morph = cv2.morphologyEx(np.array(out_light), cv2.MORPH_OPEN, kernel)

        out_morph = np.array(out_morph)

        if block2 == 'n':
            img_treated = superimpose_mask_on_image(out_morph, img, invert=False)
        elif block2 == 'y':
            img_treated = superimpose_mask_on_image(out_morph, img, invert=True)

        ax.imshow(img_treated)

        del out_light, kernel, out, out2

        # embed()
        fg.canvas.draw()
        happy = input('Are you happy? Y/N\n')

    # img_treated = superimpose_mask(out_morph, img)
    # ax.imshow(img_treated)

    num_pix = len(out_morph[out_morph == 255])

    surface_area = num_pix * px_size[0] * px_size[1]

    msk = list((str(np.where(out_morph == 255)[0]), str(np.where(out_morph == 255)[1])))

    store, store_short = store_area(store, store_short, file, msk, surface_area, parallel=count)

    ax.set_title('Do you wish to mark bees on the picture? Y or N')
    mark_yesno = input('Do you wish to mark bees on the ? Y or N\n')

    if mark_yesno == 'y':
        ax.imshow(img)
        fg.canvas.draw()
        bees = mark_bees(fg, ax)
        ax.plot(bees[:, 0], bees[:, 1], linestyle='', color='r', marker='.')
        store, store_short = store_features(store, store_short, file, bees, surface_area, parallel=count)
        ax.imshow(img_treated)
        ax.plot(bees[:, 0], bees[:, 1], linestyle='', color='r', marker='.')

    return img_treated, num_pix, msk, fg, ax, store, store_short


if __name__ == '__main__':

    """ declare myself """

    print("=============================================================")
    print("|    Measure features - Simple tool for simple jobs v.1.1   |")
    print("|                                                           |")
    print("|                   modul bee_count                         |")
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

        """ stretch """
        # new_image, im_index, dimensions, pix_size, ax2 = stretch_comb(filename, fig, ax2)

        """ action """
        """ main loop """
        stay = True
        while stay:
            ax2.imshow(imago)
            fig.canvas.draw()
            ax2.set_title('Now what? (S)tretch & calibrate, (M)easure area, (A)utomatically measure area, Measure (L)ength, (C)lose image')
            now_what = input(
                'Now what? (S)tretch & calibrate, (M)easure area, (A)utomatically measure area, Measure (L)ength, (C)lose image\n')

            if now_what == 's':
                # fig.canvas.manager.window.deiconify()
                # fig.canvas.manager.window.tkraise()
                imago, im_index, dimensions, pix_size, fig, ax2 = stretch_comb(filename, fig, ax2)

            if now_what == 'm':
                # fig.canvas.manager.window.deiconify()
                # fig.canvas.manager.window.tkraise()
                storage, storage_short, counter, fig, ax2 = select_area(title_fn, fig, ax2, storage, storage_short,
                                                                        im_index, pix_size, counter)
            if now_what == 'a':
                # fig.canvas.manager.window.deiconify()
                # fig.canvas.manager.window.tkraise()
                imago, pix, max, fg, ax, storage, storage_short = get_bee_coverage(title_fn, imago, fig, ax2, pix_size,
                                                                                storage, storage_short)
                surface = pix*pix_size[0]*pix_size[1]

                pass

            elif now_what == 'l':
                # fig.canvas.manager.window.deiconify()
                # fig.canvas.manager.window.tkraise()
                storage, storage_short, counter, fig, ax2 = measure_distance(title_fn, fig, ax2, storage, storage_short, pix_size,
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
