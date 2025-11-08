import os
import argparse
import torch
import numpy as np
from shutil import rmtree
from math import log10
from PIL import Image
from torchvision.transforms import ToTensor

import scripts.file_process as fp


# Progress Bar that shows how much of the process is done
def progressBar(index, total):
    fill = '█'
    percentage = ("{0:.1f}").format(100 * (index/float(total)))
    complen = int(100 * index // total)
    bar = fill * complen + '-' * (100 - complen)
    print('\r|%s| %s%% %s' % (bar, percentage,"Complete!"), end='\r')
    if index == total:
        print()


# Asks user if there exists a subdirectory within zip that contains images
def dir_in_dir(path):
    print("Does the zip include images on root or inside folder?")
    check = input("Type 1 for yes, 0 for no: ")
    if check:
        ext_path = input("Type the path until images are seen:\n" )
        return path, ext_path
    else:
        return path, None


# Asks user about the path of the dataset
def batch_dir_input(root_in):
    while True:
        temp = input("Type in the relative path of data set: ")
        test = os.path.join(root_in, temp)
        print("Path: " + str(test))
        if os.path.isfile(test):
            train_path = test
            break
        else:
            print("Path doesn't end with a file!")
    extr_path, folder_path = dir_in_dir(train_path)
    return extr_path, folder_path


def main():
    # asks for LR files that will be upscaled
    print("\nLow-Res files zip input")
    print("------------------")
    root_dir = os.path.realpath('.')
    batch_dirs = batch_dir_input(root_dir)
    batch_dir = fp.prep_files(root_dir, batch_dirs[0], batch_dirs[1], "comparison")

    # asks for HR counterparts of LR files to be compared
    print("\nHigh-Res versions of LR files zip input")
    print("---------------------------------")
    origin_dirs = batch_dir_input(root_dir)
    origin_dir = fp.prep_files(root_dir, origin_dirs[0], origin_dirs[1], "origin")

    i = 0
    tot = len(os.listdir(origin_dir))

    for origin_img in os.scandir(origin_dir):
        i += 1
        progressBar(i,tot)

        # conversion of original image to YCbCr format and y part is stored in a np.array for calculations
        to_compare = Image.open(origin_img.path).convert('YCbCr')
        y_org, _, _ = to_compare.split()
        y_org = np.array(y_org)

        # assuming that LR file shares the same name with HR file
        # searches for a file with matching name
        for x in os.scandir(batch_dir):
            if origin_img.name[:-4] in x.name:
                image = x
                break

        # after finding the LR counterpart, converts it into YCbCr format for
        # upscaling and calculation purposes
        to_upscale = Image.open(image.path).convert('YCbCr')
        width, height = to_upscale.size
        to_upscale = to_upscale.resize((width*3, height*3), resample=Image.BICUBIC)
        y, cb, cr = to_upscale.split()
        y_comp = np.array(y)


        # creates a log file within the given path and stores PSNR value accordingly
        f1 = open("PSNR_bicubic_logs.log", 'a+')
        mse = np.mean((y_org - y_comp) **2)
        if mse == 0:
            psnr = 100
        else:
            PIXEL_MAX = 255.0
            psnr = 20 * log10(PIXEL_MAX / np.sqrt(mse))
        f1.write("PSNR for " + image.name[:-4] + ": " + str(psnr) + "\n")
        f1.close()

    temp = os.path.join(root_dir, "extr")
    rmtree(temp, ignore_errors=True)
    print("SR task is done!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DRRA Project")
    parser.add_argument('--cuda', action='store_true', help="enable cuda?")
    args = parser.parse_args()

    print("Input args:" + str(args))

    main()
