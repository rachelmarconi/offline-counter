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
import matplotlib.pyplot as plt

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

    for i in range(45):  # was 45
        _, im = cVideo.read()
        # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('image', gray)
        frameStruct['frameStore'][:, :, i] = im[:, :, 0]
        cCount.append(np.uint8(0)) # no count for these frames I hope

        ######### run model
    backgroundModel = np.mean(frameStruct['frameStore'][:, :, :], axis=2)

    bA = myBlobAnalyzerSide()

    bA.percentFrameRemoveX = [1 / 100.0, 44 / 100.0]  # [0,0]#
    bA.percentFrameRemoveY = [2 / 100.0, 2 / 100.0]  # [0,0]#
    flipHer = True
    maxBlob = 3400.0  # 5800, 3400, 2800, 1280, 1200
    minBlob = maxBlob * 0.35


    frame = (255. - backgroundModel).astype('uint8')
    if flipHer:
        frame = np.flip(np.flip(frame, 0), 1)

    # mcf debug
    # bA.show_frame_image(frame,"Background (255-frame)")
    # cv2.waitKey(-1)

    #print("frame = np.array(", frame.tolist(), ")")
    gray_cutoff = 120
    frame[frame < gray_cutoff] = 0
    frame[frame >= 55] = 1

    bA.bkmask = cv2.dilate(frame, None)

    # mask = frame.astype('int16')
    # # mask = imfill(mask);
    # mask = cv2.GaussianBlur(mask, (3, 3), 0)
    # frame = mask.astype('bool')

    # cImg, sideView = bA.CropImage(frame, True)
    #
    # contours, sidecontours, hier = bA.get_contours(cImg, sideView)
    #
    # bA.avoidSpots = []
    # minSpot = 5   # Minimum area size to avoid
    # bigSpot = 150  # Too big might cause issues
    # dirty_main = False
    # dirty_side = False
    #
    # print("Main spots: ")
    # for i in range(len(contours)):
    #     a = cv2.contourArea(contours[i])
    #     r = cv2.boundingRect(contours[i])
    #     #print("\tArea: {:.2f} Location: {}".format(a, r))
    #     if a > minSpot:
    #         bA.avoidSpots.append(r)
    #     if a > bigSpot:
    #         dirty_main = True
    # print("Side spots: ")
    # for i in range(len(sidecontours)):
    #     a = cv2.contourArea(sidecontours[i])
    #     r = cv2.boundingRect(sidecontours[i])
    #     #print("\tArea: {:.2f} Location: {}".format(a, r))
    #     if a > minSpot:
    #         bA.avoidSpots.append(r)
    #     if a > bigSpot:
    #         dirty_side = True
    #
    # if dirty_main:
    #     print("***** Large spot in the main window *****" )
    # if dirty_side:
    #     print("***** Large spot in the side window *****" )

    bA.concavityThresh = 8#18#8#18#32
    bA.maxBlobSize = 0
    bA.minBlobAreaAbs = 300;
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
    while isValid:
        # _,cFrame = cVideo.read()
        iFrame = iFrame + 1

        print("Frame #", iFrame)
        # mcf debug
        # if iFrame == 1075:
        #     bA.show_frame_image(backgroundModel, "Background (Not flipped)")
        #     cv2.waitKey(-1)
        #     plt.title("Foreground (Not flipped)")
        #     plt.imshow(cFrame)
        #     plt.show()
        #     cv2.waitKey(-1)

        frame = abs(backgroundModel - cFrame[:, :, 0]).astype('uint8')
        if flipHer:
            frame = np.flip(np.flip(frame, 0), 1)

        # mcf debug
        # if iFrame == 1075:
        #     bA.show_frame_image(frame, "Difference (flipped)")
        #     cv2.waitKey(-1)

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
    if infile != None:
        runSingleVideo(infile)
    cv2.destroyAllWindows()
    # for vFileName in videoFileList:
    # print(vFileName)
    # runSingleVideo(vFileName)
