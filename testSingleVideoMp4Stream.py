# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:35:30 2019

@author: joeca
"""

from myBlobAnalyzerSide import myBlobAnalyzerSide
import numpy as np
import tracker
import cv2
import easygui
import time


def runSingleVideo(videoFileName):
    start = time.time()

    print(videoFileName)

    # New .npy file name
    npyfile = videoFileName.replace(".mp4", "_info.npy")
    cCount = []  # holds counts for each frame.

    cVideo = cv2.VideoCapture(videoFileName)
    numTotalFrames = int(cVideo.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("dimensions: " + str(h) + ", " + str(w))

    # mcf: Keep sum model for future trash compensation
    frame_sum = np.zeros([h, w], dtype=np.uint16)
    frame_keep = []
    avg_count = 25  # 45
    for i in range(avg_count):
        _, im = cVideo.read()
        frame_sum += im[:, :, 0]
        frame_keep.append(im[:, :, 0])
        cCount.append(np.uint8(0))  # no count for these frames I hope
    backgroundModel = frame_sum / avg_count

    bA = myBlobAnalyzerSide()

    bA.percentFrameRemoveX = [1 / 100.0, 44 / 100.0]  # [0,0]#
    bA.percentFrameRemoveY = [2 / 100.0, 2 / 100.0]  # [0,0]#
    flipHer = False
    maxBlob = 3400.0  # 5800, 3400, 2800, 1280, 1200
    minBlob = maxBlob * 0.35

    frame = (255. - backgroundModel).astype('uint8')
    if flipHer:
        frame = np.flip(frame)

    gray_cutoff = 120
    frame[frame < gray_cutoff] = 0
    frame[frame >= 55] = 1

    bA.bkmask = cv2.dilate(frame, None)

    bA.concavityThresh = 8  # 18#8#18#32
    bA.maxBlobSize = 0
    bA.minBlobAreaAbs = 300
    bA.minBlobAreaPercentage = 35
    bA.minBlobArea = bA.minBlobAreaAbs

    bA.sideViewSizeAdjust = 50  # 75

    estVelStart = np.array([[0, 9]], 'double')  # [0,50].*(120/FPS).*(fliplr(size(frame)./[480,720]));

    maxAssign = 115  # 65#110#120#65

    tracks = []

    nextId = 1  # ID of the next track

    time2Run = numTotalFrames

    countVec = np.zeros((time2Run, 1))
    numTracks = np.zeros((time2Run, 1))

    curId = 1
    numFrames = 1
    isValid, cFrame = cVideo.read()
    iFrame = 0

    # mcf: Create background output file
    # out_file = videoFileName[:-4] + "_bkg.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter(out_file, fourcc, 20.0, (w, h))
    max_ms = 0
    while isValid:
        step_start = time.time()
        # _,cFrame = cVideo.read()
        iFrame = iFrame + 1

        # mcf: background mask
        frame_sum -= frame_keep[0]  # Remove the old
        frame_sum += cFrame[:, :, 0]  # add the new
        frame_keep.pop(0)
        frame_keep.append(cFrame[:, :, 0])
        bk_frame = (255. - (frame_sum / avg_count)).astype('uint8')

        # mcf: debug create moving average video
        # image = bk_frame.copy()
        # image = np.expand_dims(image, axis=2)
        # image = np.concatenate((image, image, image), axis=2)
        # out.write(image)

        bk_frame[bk_frame < 100] = 0
        bk_frame[bk_frame >= 1] = 1

        frame = abs(backgroundModel - cFrame[:, :, 0]).astype('uint8')
        if flipHer:
            frame = np.flip(frame)
            # mcf background mask
            bk_frame = np.flip(bk_frame)

        kernel = np.ones((4, 4), np.uint8)
        bA.bkmask = cv2.dilate(bk_frame, kernel, iterations=1)

        frame[frame < gray_cutoff] = 0
        frame[frame >= 55] = 1

        tracksNew, nextId = tracker.step(frame, bA, tracks, maxAssign, curId, estVelStart,
                                         numFrames, False, maxBlob, minBlob)
        cCount.append(np.uint8(nextId - 1))

        tracks = tracksNew.copy();

        numTracks[i] = len(tracksNew)

        if (nextId - curId) != 0:
            print("pill " + str(curId) + " found at " + str(iFrame))
            countVec[i] = curId

        curId = nextId

        # if bA.frameCount >=50:
        # break
        isValid, cFrame = cVideo.read()

        ms = (time.time() - step_start) * 1000
        if max_ms < ms:
            max_ms = ms
        print('frame time:{:.4}ms'.format(ms))

    # mcf: debug, close background test video file
    # out.release()
    print('Maximum step time:{:.4}ms'.format(max_ms))
    s = (time.time() - start)
    print('file time:{:.4}s'.format(s))

    print(f'min blob: {bA.minBlobArea}')
    print(f'max blob: {bA.maxBlobSize}')
    print("count: " + str(curId - 1))
    if bA.floating_debris:
        print("Possible miscount, moving debris detected.")

    while len(cCount) <= numTotalFrames:
        cCount.append(np.uint8(nextId - 1))

    np.save(npyfile, cCount)


if __name__ == "__main__":

    infile = easygui.fileopenbox(msg='Please locate the video file',
                                 title='Specify File', default='transferVideos/*.mp4',
                                 filetypes='*.mp4')

    if infile != None:
        runSingleVideo(infile)
    cv2.destroyAllWindows()
