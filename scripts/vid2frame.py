import sys, os, glob
import argparse

import cv2

def extractImages(pathIn, pathOut, img_format='.png'):

    os.makedirs(pathOut, exist_ok=True)
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(pathOut + '/frame%04d' % count + img_format, image)     
        success,image = vidcap.read()
        count += 1

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video", required=True)
    a.add_argument("--pathOut", help="path to images", required=True)
    args = a.parse_args()
    print(args)

    # single video
    extractImages(args.pathIn, args.pathOut)
        