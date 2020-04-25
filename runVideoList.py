# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:35:30 2019

@author: joeca
"""
import os

from myBlobAnalyzerSide import myBlobAnalyzerSide
import numpy as np
import tracker
import cv2
import csv
import sys
from os import path
from pathlib import Path

results_file = open("results.txt", "w")

def runSingleVideo(videoFileName, max_blob = 2800, flip = False, good_count = 200):
    #print(videoFileName)
    global passed_count, failed_count
    #New .npy file name
    npyfile = videoFileName.replace(".mp4", "_info.npy")
    cCount = [] # holds counts for each frame.
    cVideo = cv2.VideoCapture(videoFileName)

    # Set stdio to log file for the run
    stdout_fileno = sys.stdout
    log_path = path.join('.', 'logs')
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    p = path.join(log_path, Path(videoFileName).stem + ".txt")
    sys.stdout = open(p, 'w')

    numTotalFrames = int(cVideo.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("dimensions: " + str(h) + ", " + str(w))
    frameStruct = {'frameStore': np.zeros((h, w, 45))}

    for i in range(45): # was 45
        _, im = cVideo.read()
        frameStruct['frameStore'][:, :, i] = im[:, :, 0]
        cCount.append(np.uint8(0)) # no count for these frames I hope

        ######### run model
    backgroundModel = np.mean(frameStruct['frameStore'][:, :, :], axis=2)

    bA = myBlobAnalyzerSide()

    bA.percentFrameRemoveX = [1 / 100.0, 44 / 100.0]  # [0,0]#
    bA.percentFrameRemoveY = [2 / 100.0, 2 / 100.0]  # [0,0]#

    # This was added to create a background mask
    gray_cutoff = 150
    frame = (255. - backgroundModel).astype('uint8')
    if flip:
        frame = np.flip(np.flip(frame, 0), 1)

    frame[frame < gray_cutoff] = 0
    frame[frame >= 55] = 1

    bA.bkmask = cv2.dilate(frame, None)

    bA.concavityThresh = 8#18#8#18#32
    bA.maxBlobSize = max_blob
    bA.minBlobAreaAbs = 300
    bA.minBlobAreaPercentage = 35
    bA.minBlobArea = bA.minBlobAreaAbs

    bA.sideViewSizeAdjust = 50  # 75

    estVelStart = np.array([[0, 9]], 'double')  # [0,50].*(120/FPS).*(fliplr(size(frame)./[480,720]));

    maxAssign = 115  # 65#110#120#65
    # 1280.0 3400.0 2800.0
    maxBlob = max_blob
    minBlob = maxBlob * 0.35
    tracks = []
    flipHer = flip
    nextId = 1  # ID of the next track
    time2Run = numTotalFrames

    countVec = np.zeros((time2Run, 1))
    numTracks = np.zeros((time2Run, 1))

    curId = 1
    numFrames = 1

    isValid, cFrame = cVideo.read()
    iFrame = 0
    while isValid:
        # _,cFrame = cVideo.read()
        iFrame = iFrame + 1

        # frame = abs(backgroundModel - cFrame[:,:,0]).astype('uint8')
        frame = abs(backgroundModel - cFrame[:, :, 0]).astype('uint8')
        if flipHer:
            frame = np.flip(np.flip(frame, 0), 1)

        frame[frame < gray_cutoff] = 0
        frame[frame >= 55] = 1

        tracksNew, nextId = tracker.step(frame, bA, tracks, maxAssign, curId, estVelStart, numFrames, False, maxBlob,
                                         minBlob)
        cCount.append(np.uint8(nextId-1))

        tracks = tracksNew.copy();

        numTracks[i] = len(tracksNew)

        if (nextId - curId) != 0:
            print("pill " + str(curId) + " found at " + str(iFrame))
            countVec[i] = curId

        curId = nextId

        # if bA.frameCount >=50:
        # break
        isValid, cFrame = cVideo.read()

    # print this to log file
    print(f'min blob: {bA.minBlobArea}')
    print(f'max blob: {bA.maxBlobSize}')
    print('Flip: ', flipHer)
    print("count: " + str(curId - 1))

    sys.stdout.flush()
    sys.stdout.close()

    # print set to console
    sys.stdout = stdout_fileno


    print(f'min blob: {bA.minBlobArea}')
    print(f'max blob: {bA.maxBlobSize}')
    print('Flip: ', flipHer)
    print("count: " + str(curId - 1))

    while len(cCount) <= numTotalFrames:
        cCount.append(np.uint8(nextId - 1))

    np.save(npyfile, cCount)

    if (curId - 1) == good_count:
        results_file.write("Passed expected {} counted {}\n".format(good_count, (curId-1)))
        print("Passed expected {} counted {}\n".format(good_count,(curId-1)))
        passed_count +=1
    else:
        results_file.write("Failed expected {} counted {}\n".format(good_count, (curId-1)))
        print("Failed expected {} counted {}\n".format(good_count, (curId-1)))
        failed_count += 1

if __name__ == "__main__":
    global passed_count, failed_count

    passed_count = 0
    failed_count = 0
    flip = False

    max_size = 0
    pill_count = 0

    csvfile = "videos.csv"
    csvDataFile = open(csvfile)
    readCSV = csv.reader(csvDataFile, delimiter=',')
    if readCSV is None:
        print("Main list file not found: ", csvfile)
        exit()

    for row in readCSV:
        if len(row) < 4:
            #print("CSV incomplete row")
            continue
        elif not path.exists(row[0]):
            print("File not found: ", row[0])
            continue

        flip = True if "true" in row[1].lower() else False

        try:
            max_size = int(row[2])
            pill_count = int(row[3])
        except:
            continue

        results_file.write(row[0] + "\n")
        print('Running: ', row[0])
        runSingleVideo(row[0], max_size, flip, pill_count)

    results_file.write("Videos run = {}\n".format(passed_count + failed_count))
    results_file.write("Total passed = {}\n".format(passed_count))
    results_file.write("Total failed = {}\n".format(failed_count))
    # show results on console
    print("***** COMPLETE *****")
    print("Videos run = {}".format(passed_count+failed_count) )
    print("Total passed = {}".format(passed_count) )
    print("Total failed = {}".format(failed_count) )

    results_file.close()