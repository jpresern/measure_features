#!/usr/bin/env python3

import tkinter as tk
import matplotlib

matplotlib.use("TkAgg")

import sys, os, glob
import re
import pandas as pd
import numpy as np
import math
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

from measure_features import get_things_opened, bulk_measure_ci, image_display, prepare_storage,\
    prepare_short_storage


__author__ = "bpiskur, jpresern"


def get_folder_opened():
    idir = './samples'
    window_open = tk.Tk()
    window_open.withdraw()
    im_name = askdirectory(initialdir=idir, title='Select image file')
    window_open.destroy()

    return im_name


def get_bulk_saved(store, store_short, dir):
    """ get filename """
    idir = dir.split('/')[-1]
    window_save = tk.Tk()
    window_save.withdraw()
    save_fn = dir + '/' + idir

    window_save.destroy()

    store.to_csv(save_fn + '.csv', index_label="index")
    store_short.to_csv(save_fn + '_short.csv', index_label="index")
    """ resets instructions to nothing before save"""


if __name__ == '__main__':

    """ declare myself """
    #TODO: implement imutils.skeletonize
    print("=============================================================")
    print("|    Measure features - Simple tool for simple jobs v.1.1   |")
    print("|                                                           |")
    print("|                          modul CI                         |")
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

        directory = get_folder_opened()

        wings = []

        if not wings:
            wings = glob.glob(directory + '/*.bmp')
        elif not wings:
            wings = glob.glob(directory + '/*.JPG')
        elif not wings:
            wings = glob.glob(directory + '/*.tif')
        else:
            print('No files found')

        """ prepare empty storage """
        storage = prepare_storage()
        storage_short = prepare_short_storage()
        counter = 0

        for wing in wings:
            """ sort extensions etc """
            fn = wing.rsplit('.', maxsplit=1)[0]
            ext = wing.rsplit('.', maxsplit=1)[1]
            ext = '.' + ext
            title_fn = wing.rsplit('/', maxsplit=1)[-1]

            """ load image, load meta data, display image and calibrate """
            fig, ax2, im_index, pix_size = image_display(wing, fn)

            """ action """
            """ main loop """

            fig.canvas.manager.window.deiconify()
            fig.canvas.manager.window.tkraise()
            storage, storage_short, counter = bulk_measure_ci(title_fn, fig, ax2, storage, storage_short, pix_size,
                                                         counter)
            plt.close(fig)
            counter += 1
        get_bulk_saved(storage, storage_short, directory)
        und_jetzt = input('And now what? (L)oad new samples, or (Q)uit?\n')
        if (und_jetzt == 'q') or (und_jetzt == 'Q'):
            do_work = False

    exit()
