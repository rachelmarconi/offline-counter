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
        self.minBlobAreaAbs = 100
        self.minBlobArea = self.minBlobAreaAbs
        self.minBlobAreaPercentage = 0
        self.concavityThresh = 8
        self.sideLookTresh = .30

        self.percentFrameRemoveX = [.06, .47]
        self.percentFrameRemoveY = [.4, .04]
        self.rightEdge = 140

        self.sideViewStart = 190
        self.sideViewEnd = 255
        self.sideViewSizeAdjust = 100  # in percent
        self.side_contours = None

        self.areaVec = np.zeros((500, 1))
        self.minAreaVec = np.nan * np.ones((500, 1))

        self.maxBlobSize = 800
        self.clearEst = 0
        self.frameCount = 1
        self.blobSplit = 0
        self.maxAssign = 50
        self.stepCount = 0
        self.first = True

        # To find concavity indents
        # A valid concavity slope vector will exceed this threshold
        self.thresholdSlopeVect = 65
        # Take a point every (skipPts) around contour perimeter.
        self.skipPts = 10 # take every 10th point

    def step(self, cImgIn, predictedCentroidsList, garbMaxAssign, calibrate, inputMaxBlob, inputMinBlob):
        if not calibrate:
            self.maxBlobSize = inputMaxBlob
            self.minBlobArea = inputMinBlob
            # Scale the max concavity based on the pill size
            # self.concavityThresh = 7+int(self.maxBlobSize / 1200) # ((self.maxBlobSize / 3.1415926)**.5) / 2.6
            self.concavityThresh = ((self.maxBlobSize / 3.1415926)**.5) / 2.6
        self.maxAssign = self.maxBlobSize ** .5
        self.stepCount += 1
        area = np.empty(0)
        centroids = np.empty((0, 2))
        isFirst = True

        cImg, sideView = self.CropImage(cImgIn, self.first)
        self.first = False

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

            tmpGarbage, contoursAll, hierarchy = cv2.findContours(sideView, cv2.RETR_TREE,
                                                                  1)  # changed from ...(tmpcImg,2,1)
            self.side_contours = [contoursAll[i] for i in range(len(contoursAll)) if hierarchy[:, i, -1] == -1]

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
            # self.find_nearest_predicted_area(contourAreas, contourCentroids, contours, matchedCentroids,
            #                                  predictedCentroidsList)

            # then go through contours again for adding centroids to final list
            area, centroids = self.get_centroids_from_areas(area, centroids, contourAreas, contourCentroids, contours,
                                                            isFirst, labelVec, matchedCentroids, numDiffPillsinConts,
                                                            bottomEdge, topEdge)

            if self.sideViewSizeAdjust != 0 and self.frameCount <= 5:
                # add in sideview areas
                side_areas = np.array(self.getSideAreas(sideView))
                area = np.append(area, side_areas)

            # print('time to check:  {0:.06f}'.format(time.time() - startEval))
            if area.size > 0:
                # if we've seen 50 (500 frames of pills) pills, really no need to continue computation
                if (calibrate and self.frameCount <= 500 and area.size > 0):
                    self.maxBlobSize = np.max(np.append(area, self.maxBlobSize))  # take the max
                    self.areaVec[self.frameCount - 1] = np.max(area)  # max of the ones in this frame
                    self.maxBlobSize = np.percentile(self.areaVec[0:self.frameCount],
                                                     98)  # 98th percentile (reduce some outliers)
                    self.minBlobArea = np.max(
                        ((self.minBlobAreaPercentage / 100.0) * self.maxBlobSize, self.minBlobAreaAbs))
                self.frameCount = self.frameCount + 1  # increase framecount

                if (self.frameCount <= 50) and hierarchy is not None:  # 50 frames of pills to check clear est
                    self.clearEst = self.clearEst + int(
                        (len(hierarchy[0]) - len(contours)) > 1)  # create an estimate for how clear it is
        return area, centroids

    def get_centroids_from_areas(self, area, centroids, contourAreas, contourCentroids, contours, isFirst, labelVec,
                                 matchedCentroids, numDiffPillsinConts, bottomEdge, topEdge):
        """*************************************************************************************************************
        * This method will match the centroids to the found areas and attempt to set or adjust the centroid position
        * to its most probable location.
        """
        SIDE_VIEW_OFFSET = 15  # approximat pixel error in side view image
        for iObj in labelVec:
            foundMatches = np.zeros((0, 2))  # none
            cArea = contourAreas[iObj]
            estPills = 1
            if cArea > self.minBlobArea:
                cCont = contours[iObj]
                x, y, w, h = cv2.boundingRect(cCont)
                if  cArea <= (self.maxBlobSize * 1.1):
                    # add actual centroid
                    print("using area centroid")
                    foundMatches = np.vstack((foundMatches, contourCentroids[iObj]))
                else: # cArea > (self.maxBlobSize * 1.1):
                    if ((y + h) < bottomEdge and y > topEdge) or numDiffPillsinConts[iObj] > 1:
                        x1 = contourCentroids[iObj][0]
                        y1 = contourCentroids[iObj][1]
                        ct1, ct2 = self.find_2_blob_centroids(cCont, x1, y1)
                        foundMatches = np.vstack((foundMatches, ct1))
                        foundMatches = np.vstack((foundMatches, ct2))
                        estPills = int(cArea / (self.maxBlobSize + 0.1))
                    else:
                        foundMatches = np.vstack((foundMatches, contourCentroids[iObj]))

                # comes from concavity defect counter
                if estPills < numDiffPillsinConts[iObj]:
                    estPills = numDiffPillsinConts[iObj]

                # check side view count
                # no side view check if others at this y range
                others = False
                y_tolarance = 2  # Alowable variance in sideview y value
                for i in labelVec:
                    if(i != iObj):
                        x2, y2, w2, h2 = cv2.boundingRect(contours[i])
                        #if (y2 >= y and y2 <= y+h) or (y2 < y and y2+h2 > y):
                        # disqualified only if another pill is totally withing our latitude
                        if y2 >= y - 6 and y2 + h2 <= y + h + 6:
                            others = True # other pills are in our Y range
                            break
                if not others and self.side_contours is not None:
                    print("Getting side count pill: left = {}, top = {}, bottom = {}".format(x, y, y+h))
                    side_count = self.get_side_count(y - y_tolarance,
                                                     y + h + y_tolarance, bottomEdge, topEdge)
                    if estPills < side_count:
                        estPills = side_count
                        print("Using side count")

                while estPills > len(foundMatches):
                    ct = contourCentroids[iObj]
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

    def find_2_blob_centroids(self, cCont, x, y):
        """*************************************************************************************************************
        * This method will return the two best guess centroids for a multi-pill blob.
        """
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

    def count_pills_in_cont(self, cont, bottomEdge, topEdge):
        """*************************************************************************************************************
        * This method will get the estimated pill count for a given contour based on concavities.
        """
        cHull = cv2.convexHull(cont, returnPoints=False)
        defects = cv2.convexityDefects(cont, cHull)
        if defects is not None:
            x, y, w, h = cv2.boundingRect(cont)
            # contYArray = np.squeeze(np.array(cCont[cHull]))[:, 1]
            # if np.any(contYArray >= bottomEdge) or np.any(contYArray <= topEdge):

            # MCF, Use defect concavity point to determine if defect is safe to use for pill split
            nDiffs = 1
            keep = 0
            for i in range(len(defects[:,:,3])):
                if(defects[i][0][3] >= self.concavityThresh * 256 ):
                    # Get the X and Y arrays from every (self.skipPts) pixel in the contour
                    xarray = np.squeeze(cont[0::self.skipPts])[:, 0]
                    yarray = np.squeeze(cont[0::self.skipPts])[:, 1]
                    # Get the slopes delta x, delta y around the perimeter of the contour
                    dx = np.diff(xarray)
                    dx = np.append(dx, [dx[0]])  # Wrap
                    dy = np.diff(yarray)
                    dy = np.append(dy, [dy[0]])  # Wrap
                    for i in range(len(dx) - 1):
                        val = (dx[i] * dy[i + 1] - dy[i] * dx[i + 1])
                        if val > self.thresholdSlopeVect and yarray[i+1] > topEdge + 4 \
                                and yarray[i+1] + self.maxAssign < bottomEdge:
                            keep +=1
                    break
            if keep > 0:
                nDiffs = int(np.ceil(keep/2.0)+1)
        else:
            nDiffs = 1

        return nDiffs

    def find_centroids_and_areas(self, cImg, labelVec, contourAreas, contourCentroids, contours, numDiffPillsinConts,
                                 bottomEdge, topEdge):
        """*************************************************************************************************************
        * This method will search through the contours and add contourCentroids for appropriate contours.
        """
        for iObj in labelVec:
            cCont = contours[iObj]
            nDiffs = self.count_pills_in_cont(cCont, bottomEdge, topEdge)

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

    def CropImage(self, cImgIn, first):
        """***************************************************************************************
        * This method will return image frames for the main and side view with the unused
        * portions of each image zeroed out.
        * based on the parameters percentFrameRemoveX and percentFrameRemoveY.
        """
        # make this a percentage of the frame
        cImg = cImgIn.copy()
        # Set the left percent(0.01) cols to zero
        left_x = int(np.ceil(cImg.shape[1]*(self.percentFrameRemoveX[0] )))
        cImg[:, 0:int(np.ceil(cImg.shape[1] * self.percentFrameRemoveX[0]))] = 0
        # Set the right percent(0.44) cols to zero
        right_x = int(cImg.shape[1] - (np.ceil(cImg.shape[1] * self.percentFrameRemoveX[1])))
        cImg[:, -int(np.ceil(cImg.shape[1] * self.percentFrameRemoveX[1])):] = 0
        # Set top percent(0.02) rows in cImg to zeros
        top_y = int(np.ceil(cImg.shape[0]*self.percentFrameRemoveY[0]))
        cImg[0:int(np.ceil(cImg.shape[0] * self.percentFrameRemoveY[0])), :] = 0
        # Set the bottom percent(0.02) rows to zero
        bottom_y = int(cImg.shape[0] - (np.ceil(cImg.shape[0]*self.percentFrameRemoveY[1])))
        cImg[-int(np.ceil(cImg.shape[0] * self.percentFrameRemoveY[1])):, :] = 0

        if first:
            print("Image size w = {}, h= {}".format(cImg.shape[1], cImg.shape[0]))
            print("Main window mask from {},{} to {},{}".format(left_x,top_y,right_x,bottom_y))

        # MAKE side view
        sideView = cImgIn.copy();
        # Set the right most percent(0.01) cols to zero
        right_x = int(cImg.shape[1] - (np.ceil(cImg.shape[1] * self.percentFrameRemoveX[0])))
        sideView[:,-int(np.ceil(cImg.shape[1] * self.percentFrameRemoveX[0]))] = 0
        # Set the left percent(44% + 25%) cols to zero
        left_x = int(np.ceil(cImg.shape[1]*(self.percentFrameRemoveX[1] + .20)))
        sideView[:,0:int(np.ceil(cImg.shape[1]*(self.percentFrameRemoveX[1] + .20)))] = 0
        # Set top percent(0.02) rows in cImg to zeros
        sideView[0:int(np.ceil(cImg.shape[0]*self.percentFrameRemoveY[0])),:] = 0
        # Set the bottom percent(0.02) rows to zero
        sideView[-int(np.ceil(cImg.shape[0]*self.percentFrameRemoveY[1])):,:] = 0
        sideView = 255*sideView.copy().astype('uint8')
        if first:
            print("Side window mask from {},{} to {},{}".format(left_x,top_y,right_x,bottom_y))

        return cImg, sideView

    def getSideAreas(self, sideView):
        """***********************************************************************************************
        * This method returns qualified contour areas from the side view image.
        """
        side_areas = []
        for cCont in self.side_contours:
            cArea = cv2.contourArea(cCont)
            if cArea > self.minBlobArea:
                side_areas.append(cArea * self.sideViewSizeAdjust / 100.0)
        return side_areas

    def get_side_count(self, top, bottom, bottomEdge, topEdge):
        """*********************************************************************************************
        * This method returns number of estimated pills in the side view between top and bottom.
        """
        num_counted = 0

        for cont in self.side_contours:
            area = cv2.contourArea(cont)
            if area >= self.minBlobArea:
                x, y, w, h = cv2.boundingRect(cont)
                print("Side pill: left = {}, top = {}, bottom = {}, area = {}".format(x, y, y + h, area))

                # Is this contour at the right height?
                if y >= top and (y + h) <= bottom:
                    num_counted += self.count_pills_in_cont(cont, bottomEdge, topEdge)

                # Area estimates are not reliable
                #estPills = 1 + int(area / (self.maxBlobSize + 0.1))
                #num_counted = max(num_counted, estPills)
        print("Side pill count: {}".format(num_counted))

        return num_counted