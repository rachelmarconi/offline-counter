# -*- coding: utf-8 -*-
"""
Created on Wed May 30 13:22:07 2018

@author: Joe
"""
# %%
import numpy as np
import cv2
import datetime


# %%
class myBlobAnalyzerSide(object):
    def __init__(self):
        self.minBlobAreaAbs = 100;
        self.minBlobArea = self.minBlobAreaAbs
        self.minBlobAreaPercentage = 0
        self.concavityThresh = 5;
        self.sideLookTresh = .30;

        self.percentFrameRemoveX = [.06, .47]
        self.percentFrameRemoveY = [.4, .04]
        self.rightEdge = 140

        self.sideViewStart = 190
        self.sideViewEnd = 255

        self.sideViewSizeAdjust = 100  # in percent

        self.areaVec = np.zeros((500, 1))
        self.minAreaVec = np.nan * np.ones((500, 1))

        self.maxBlobSize = 800
        self.clearEst = 0
        self.frameCount = 1
        self.blobSplit = 0
        self.maxAssign = 50
        self.stepCount = 0

    def step(self, cImgIn, predictedCentroidsList, garbMaxAssign, calibrate, inputMaxBlob, inputMinBlob):
        if not calibrate:
            self.maxBlobSize = inputMaxBlob
            self.minBlobArea = inputMinBlob
        self.maxAssign = self.maxBlobSize ** .5
        self.stepCount += 1
        area = np.empty(0)
        centroids = np.empty((0, 2))
        isFirst = True

        cImg, sideView = self.CropImage(cImgIn)

        bottomEdge = int(cImg.shape[0] * (1 - self.percentFrameRemoveY[1])) - 3
        topEdge = int(cImg.shape[0] * self.percentFrameRemoveY[0]) + 3

        if np.any(cImg):
            # print("{:.0f}/{:.0f}; {:.0f}".format(self.maxBlobSize, self.minBlobArea, self.maxAssign))
            print("Step# ", self.stepCount, " pill sz: ", self.maxBlobSize, "/", self.minBlobArea, "/", self.maxAssign)

            tmpcImg = 255 * cImg.copy().astype('uint8')
            tmpGarbage, contoursAll, hierarchy = cv2.findContours(tmpcImg, cv2.RETR_TREE,
                                                                  1)  # changed from ...(tmpcImg,2,1)
            contours = [contoursAll[i] for i in range(len(contoursAll)) if hierarchy[:, i, -1] == -1]
            labelVec = np.arange(0, len(contours))

            ####################
            matchedCentroids = [[] for i in range(len(contours))]
            contourAreas = np.zeros(len(contours))  # populate iterating contours for blob division
            contourCentroids = np.zeros(shape=(len(contours), 2))
            numDiffPillsinConts = np.zeros(len(contours), dtype=int)
            ####################

            # first, go through and make centroids and areas
            self.find_centroids_and_areas(cImg, labelVec, contourAreas, contourCentroids, contours, numDiffPillsinConts,
                                          bottomEdge, topEdge)

            print("first contourCentroids:\n{}".format(contourCentroids))
            print("contour areas: {}".format(contourAreas))

            # test each predicted area to see which area is closest to this predicted
            self.find_nearest_predicted_area(contourAreas, contourCentroids, contours, matchedCentroids,
                                             predictedCentroidsList)

            # then go through contours again for adding centroids to final list
            area, centroids = self.get_centroids_from_areas(area, centroids, contourAreas, contourCentroids, contours,
                                                            isFirst, labelVec, matchedCentroids, numDiffPillsinConts)

            if self.sideViewSizeAdjust != 0 and self.frameCount <= 5:
                # add in sideview areas
                side_areas = np.array(self.getSideAreas(sideView))
                area = np.append(area, side_areas)

            # print('time to check:  {0:.06f}'.format(time.time() - startEval))
            if area.size > 0:
                if (
                        calibrate and self.frameCount <= 500 and area.size > 0):  # if we've seen 50 (500 frames of pills) pills, really no need to continue computation
                    self.maxBlobSize = np.max(np.append(area, self.maxBlobSize))  # take the max
                    self.areaVec[self.frameCount - 1] = np.max(area)  # max of the ones in this frame
                    self.maxBlobSize = np.percentile(self.areaVec[0:self.frameCount],
                                                     98)  # 98th percentile (reduce some outliers)
                    # self.minAreaVec[self.frameCount - 1] = np.min(area) # min of the ones in this frame
                    # self.minBlobArea = np.max((self.minBlobAreaAbs,
                    #                    np.percentile(self.minAreaVec[0:self.frameCount], 2))) # 98th percentile (reduce some outliers)
                    self.minBlobArea = np.max(
                        ((self.minBlobAreaPercentage / 100.0) * self.maxBlobSize, self.minBlobAreaAbs))
                self.frameCount = self.frameCount + 1  # increase framecount

                if (self.frameCount <= 50):  # 50 frames of pills to check clear est
                    self.clearEst = self.clearEst + int(
                        (len(hierarchy[0]) - len(contours)) > 1)  # create an estimate for how clear it is

        # centroids = np.transpose(centroids)
        # print("centroids: "+str(centroids))
        return area, centroids

    """*************************************************************************************************************
    * This method will match the centroids to the found areas and attempt to set or adjust the centroid position 
    * to its most probable location.
    """

    def get_centroids_from_areas(self, area, centroids, contourAreas, contourCentroids, contours, isFirst, labelVec,
                                 matchedCentroids, numDiffPillsinConts):
        for iObj in labelVec:
            foundMatches = np.zeros((0, 2))  # none
            cArea = contourAreas[iObj]
            if cArea > self.minBlobArea:
                # print('extra')
                if not matchedCentroids[iObj] == None and len(matchedCentroids[iObj]) > 1:
                    print("using predicted centroids")
                    # add the predicted points as actual points to final centroids list
                    for i in range(len(matchedCentroids[iObj])):
                        foundMatches = np.vstack((foundMatches, matchedCentroids[iObj][i][0]))
                elif not matchedCentroids[iObj] == None and len(matchedCentroids[iObj]) <= 1 \
                        and cArea <= (self.maxBlobSize * 1.1):
                    # add actual centroid
                    print("using area centroid")
                    foundMatches = np.vstack((foundMatches, contourCentroids[iObj]))
                elif not matchedCentroids[iObj] == None and len(matchedCentroids[iObj]) <= 1 \
                        and cArea > (self.maxBlobSize * 1.1):
                    cCont = contours[iObj]
                    x = contourCentroids[iObj][0]
                    y = contourCentroids[iObj][1]
                    ct1, ct2 = self.find_2_bolb_centroids(cCont, x, y)
                    foundMatches = np.vstack((foundMatches, ct1))
                    foundMatches = np.vstack((foundMatches, ct2))

                estPills = int(cArea / (self.maxBlobSize + 0.1))
                while estPills > len(foundMatches):
                    ct = [[x, y]]
                    foundMatches = np.vstack((foundMatches, ct))

                nDiffs = 1
                if len(foundMatches) > 1:
                    nDiffs = len(foundMatches)

                    if (len(matchedCentroids[iObj]) < nDiffs):
                        area = np.append(area, cArea / nDiffs)
                        cBlobSplit = cArea / self.maxBlobSize
                        self.blobSplit = np.max([self.blobSplit, cBlobSplit])
                        print('{} blob split {}'.format(
                            datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S'),
                            cBlobSplit))
                area = np.append(area, cArea / nDiffs)
                if isFirst:
                    isFirst = False
                    centroids = np.zeros((0, 2))
                centroids = np.vstack((centroids, foundMatches))
        return area, centroids

    """*************************************************************************************************************
    * This method will return the two best guess centroids for a multi-pill blob.
    """

    def find_2_bolb_centroids(self, cCont, x, y):
        # This finds a tight fitting rotated rectangle to best fit the blob
        rect = cv2.minAreaRect(cCont)
        box = cv2.boxPoints(rect)
        x1 = box[0][0]
        y1 = box[0][1]
        x2 = box[1][0]
        y2 = box[1][1]
        x4 = box[3][0]
        y4 = box[3][1]
        if ((x1 - x2) ** 2 + (y1 - y2) ** 2) > ((x1 - x4) ** 2 + (y1 - y4) ** 2):
            vecX = (x1 - x2) / 4
            vecY = (y1 - y2) / 4
        else:
            vecX = (x1 - x4) / 4
            vecY = (y1 - y4) / 4
        ct1 = [[x - vecX, y - vecY]]
        ct2 = [[x + vecX, y + vecY]]
        return ct1, ct2

    """*************************************************************************************************************
    * This method will find the area closed to the predicted centroid location.
    """

    def find_nearest_predicted_area(self, contourAreas, contourCentroids, contours, matchedCentroids,
                                    predictedCentroidsList):
        for iPred in range(len(predictedCentroidsList)):
            # this predicted centroid
            pred = predictedCentroidsList[iPred]
            print("pred: {}".format(pred))

            cClosest = 1000
            iClosest = 0
            for iCont in range(len(contours)):
                # only if the area is big enough?
                if contourAreas[iCont] >= self.minBlobArea:
                    x = contourCentroids[iCont][0]
                    y = contourCentroids[iCont][1]
                    if contourAreas[iCont] > (self.maxBlobSize * 1.1):
                        ct1, ct2 = self.find_2_bolb_centroids(contours[iCont], x, y)
                        dist = min(((ct1[0][0] - pred[0]) ** 2 + (
                                ct1[0][1] - pred[1]) ** 2) ** .5,
                                   ((ct2[0][0] - pred[0]) ** 2 + (
                                           ct2[0][1] - pred[1]) ** 2) ** .5)
                    else:
                        dist = ((x - pred[0]) ** 2 + (
                                y - pred[1]) ** 2) ** .5
                    print("dist: {:.3f}".format(dist))
                    if dist < cClosest:
                        # print("current closest")
                        cClosest = dist
                        iClosest = iCont

            if contourAreas[iClosest] >= self.minBlobArea:
                # which is closest?
                correctCont = contours[iClosest]
                # check if within area or within maxAssign

                """
                #*********** Using area centroids
                if cClosest <= self.maxAssign:
                    if contourAreas[iClosest] > (self.maxBlobSize * 1.1):
                        ct1, ct2 = self.find_2_bolb_centroids(contours[iClosest],
                                                              contourCentroids[iCont][0], contourCentroids[iCont][1])
                        if ((ct1[0][0] - pred[0]) ** 2 + (ct1[0][1] - pred[1]) ** 2) < \
                                ((ct2[0][0] - pred[0]) ** 2 + (ct2[0][1] - pred[1]) ** 2):
                            matchedCentroids[iClosest].append(np.array(ct1))
                        else:
                            matchedCentroids[iClosest].append(np.array(ct2))
                    else:
                        matchedCentroids[iClosest].append(np.array(contourCentroids[iClosest]))
                else:
                    print("Centroid: ", pred, "no match: too far")

                 """
                #*********** Using Predicted
                # find the max and min of x and y
                minx = np.min(correctCont[:, :, 0])
                maxx = np.max(correctCont[:, :, 0])
                miny = np.min(correctCont[:, :, 1])
                maxy = np.max(correctCont[:, :, 1])
                print("closest x: ({:.1f},{:.1f}). y: ({:.1f},{:.1f}); dist: {:.3f}; centroid: {}".
                      format(minx, maxx, miny, maxy, cClosest, contourCentroids[iClosest]))
                if minx < pred[0] < maxx and miny < pred[1] < maxy and not cClosest == 1000:
                    print("track is in contour")
                    if cClosest > .35 * self.maxAssign:
                        print("far: altering 10%")
                        pred[0] = pred[0] + .1 * (contourCentroids[iClosest][0] - pred[0])
                        pred[1] = pred[1] + .1 * (contourCentroids[iClosest][1] - pred[1])
                    matchedCentroids[iClosest].append(np.array([pred]))
                # check if this predicted centroid is just outside this area
                elif cClosest < self.maxAssign:
                    print("track is near contour edge")
                    matchedCentroids[iClosest].append(np.array([[(pred[0] + contourCentroids[iClosest][0]) / 2,
                                                                 (pred[1] + contourCentroids[iClosest][1]) / 2]]))
                    # print("matched: "+str(len(matchedCentroids))+" "+str(matchedCentroids))
                else:
                    print("no match: too far")
                """"""

    """*************************************************************************************************************
    * This method will search through the contours and add contourCentroids for appropriate contours.
    """

    def find_centroids_and_areas(self, cImg, labelVec, contourAreas, contourCentroids, contours, numDiffPillsinConts,
                                 bottomEdge, topEdge):
        for iObj in labelVec:
            cCont = contours[iObj]
            nDiffs = 1
            cHull = cv2.convexHull(cCont, returnPoints=False)
            defects = cv2.convexityDefects(cCont, cHull)
            if defects is not None:
                contYArray = np.squeeze(np.array(cCont[cHull]))[:, 1]
                if np.any(contYArray >= bottomEdge) or np.any(contYArray <= topEdge):
                    nDiffs = 1
                else:
                    caveKeep = self.concavityThresh * 256 < defects[:, :, 3]  # multiply by the bit size (256)
                    nDiffs = int(np.ceil(sum(caveKeep)[0] / 2.0) + 1)
                    if nDiffs > 1:
                        print('concavity: ' + str(nDiffs))
            else:
                nDiffs = 1
            # rows, cols = cImg.shape[:2]
            [vx, vy, x, y] = cv2.fitLine(cCont, cv2.DIST_L2, 0, 0.01, 0.01)

            # Set centroid at center of the blob
            contourCentroids[iObj] = np.zeros(2)
            contourCentroids[iObj][0] = x
            contourCentroids[iObj][1] = y
            contourAreas[iObj] = cv2.contourArea(cCont)

            hullArea = cv2.contourArea(cv2.convexHull(cCont))
            thisTotalConcavityArea = hullArea - contourAreas[iObj]
            # print("\t\t\t\t\t\t\t\t\t\t\t\t\ttotal concavity area: {}".format(thisTotalConcavityArea))
            numDiffPillsinConts[iObj] = nDiffs

    """*************************************************************************************************************
    * This method will return cropped main view and side images based on the parameters percentFrameRemoveX,
    * percentFrameremoveY, and percentFrameRemoveY.
    """

    def CropImage(self, cImgIn):
        # make this a percentage of the frame
        cImg = cImgIn.copy()
        # Set top percent(0.02) rows in cImg to zeros
        cImg[0:int(np.ceil(cImg.shape[0] * self.percentFrameremoveY[0])), :] = 0
        # Set the bottom percent(0.02) rows to zero
        cImg[-int(np.ceil(cImg.shape[0] * self.percentFrameremoveY[1])):, :] = 0
        # Set the left percent(0.01) cols to zero
        cImg[:, 0:int(np.ceil(cImg.shape[1] * self.percentFrameRemoveX[0]))] = 0
        # Set the right percent(0.44) cols to zero
        cImg[:, -int(np.ceil(cImg.shape[1] * self.percentFrameRemoveX[1])):] = 0
        # MAKE side view
        sideView = cImgIn.copy();
        # Set the left percent (0.40) cols to zero
        sideView[:, 0:int(np.ceil(cImg.shape[1] * self.percentFrameRemoveY[0]))] = 0
        # Set the right percent(0.04) cols to zero
        sideView[:, -int(np.ceil(cImg.shape[1] * self.percentFrameRemoveY[1])):] = 0
        # Set top percent (0.02) rows in cImg to zeros
        sideView[0:int(np.ceil(cImg.shape[0] * self.percentFrameremoveY[0])), :] = 0
        # Set bottom percent (0.02) rows in cImg to zeros
        sideView[-int(np.ceil(cImg.shape[0] * self.percentFrameremoveY[1])):, :] = 0
        sideView = 255 * sideView.copy().astype('uint8')
        return cImg, sideView

    """*************************************************************************************************************
    * This method returns qualified contour areas from the side view image.
    """

    def getSideAreas(self, sideView):
        tmpGarbage, contoursAll, hierarchy = cv2.findContours(sideView, cv2.RETR_TREE,
                                                              1)  # changed from ...(tmpcImg,2,1)
        contours = [contoursAll[i] for i in range(len(contoursAll)) if hierarchy[:, i, -1] == -1]
        side_areas = []
        for cCont in contours:
            cArea = cv2.contourArea(cCont)
            if cArea > self.minBlobArea:
                side_areas.append(cArea * self.sideViewSizeAdjust / 100.0)
        return side_areas
