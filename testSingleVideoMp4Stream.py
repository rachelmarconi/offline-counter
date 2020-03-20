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

# %%
# frameStruct = sio.loadmat(r'saveRuns\test_81aspirin_120_11.mat')

# frameStruct = sio.loadmat(r'saveRuns\testProto_fish_01.mat')
# frameStruct = sio.loadmat(r'saveRuns\test.mat')
# cFrame = frameStruct['frameStore'][:,:,2936]


def runSingleVideo(videoFileName):
    print(videoFileName)

    #New .npy file name
    npyfile = videoFileName.replace(".mp4", "_info.npy")
    cCount = [] # holds counts for each frame.

    cVideo = cv2.VideoCapture(videoFileName)

    numTotalFrames = int(cVideo.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("dimensions: " + str(h) + ", " + str(w))
    frameStruct = {'frameStore': np.zeros((h, w, 45))}

    for i in range(45):
        _, im = cVideo.read()
        frameStruct['frameStore'][:, :, i] = im[:, :, 0]
        cCount.append(np.uint8(0)) # no count for these frames I hope

        ######### run model
    backgroundModel = np.mean(frameStruct['frameStore'][:, :, :], axis=2)

    bA = myBlobAnalyzerSide()

    bA.percentFrameRemoveX = [1 / 100.0, 44 / 100.0]  # [0,0]#
    bA.percentFrameremoveY = [2 / 100.0, 2 / 100.0]  # [0,0]#

    bA.concavityThresh = 8#18#8#18#32
    bA.maxBlobSize = 0
    bA.minBlobAreaAbs = 300;
    bA.minBlobAreaPercentage = 35
    bA.minBlobArea = bA.minBlobAreaAbs

    bA.sideViewSizeAdjust = 50  # 75

    estVelStart = np.array([[0, 9]], 'double')  # [0,50].*(120/FPS).*(fliplr(size(frame)./[480,720]));

    maxAssign = 115  # 65#110#120#65

    maxBlob = 2800.0
    minBlob = maxBlob * 0.35
    tracks = []

    nextId = 1  # ID of the next track

    time2Run = numTotalFrames

    countVec = np.zeros((time2Run, 1))
    numTracks = np.zeros((time2Run, 1))

    curId = 1
    numFrames = 1
    flipHer = False
    isValid, cFrame = cVideo.read()
    iFrame = 0
    while isValid:
        # _,cFrame = cVideo.read()
        iFrame = iFrame + 1

        # frame = abs(backgroundModel - cFrame[:,:,0]).astype('uint8')
        frame = abs(backgroundModel - cFrame[:, :, 0]).astype('uint8')
        if flipHer:
            frame = np.flip(np.flip(frame, 0), 1)

        frame[frame < 150] = 0
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

    print(f'min blob: {bA.minBlobArea}')
    print(f'max blob: {bA.maxBlobSize}')
    print("count: " + str(curId - 1))

    while len(cCount) <= numTotalFrames:
        cCount.append(np.uint8(nextId - 1))

    np.save(npyfile, cCount)

if __name__ == "__main__":
    # videoFileName = r'transferVideos/2019.09.09_17.08.26_reqInv_disp101.mp4'
    #videoFileList = glob.glob('transferVideos/*.mp4')
    #videoFileList.sort()

    infile = easygui.fileopenbox(msg='Please locate the video file',
                                      title='Specify File', default='transferVideos/*.mp4',
                                      filetypes='*.mp4')

    # for vFileName in videoFileList:
    #index = 0
    #runSingleVideo(videoFileList[index])
    runSingleVideo(infile)
    # for vFileName in videoFileList:
    # print(vFileName)
    # runSingleVideo(vFileName)
