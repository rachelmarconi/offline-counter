# -*- coding: utf-8 -*-
"""
Created on Wed May 30 13:22:07 2018

@author: Joe
"""
#%%
import numpy as np
import cv2
import datetime
#import time
#

#%%
class myBlobAnalyzerSide(object):
    def __init__(self):
        self.minBlobAreaAbs = 100;
        self.minBlobArea = self.minBlobAreaAbs
        self.minBlobAreaPercentage = 0
        self.concavityThresh = 5;
        self.sideLookTresh = .30;
  
        self.percentFrameRemoveX = [.06,.47]
        self.percentFrameRemoveY = [.4,.04]   
        self.leftEdge = 140 

        self.sideViewStart = 190
        self.sideViewEnd = 255      

        self.sideViewSizeAdjust = 100 # in percent       

        self.areaVec = np.zeros((500,1))
        self.minAreaVec = np.nan*np.ones((500,1))       

        self.maxBlobSize = 800
        self.clearEst = 0
        self.frameCount = 1
        
        self.blobSplit = 0
  
    def step(self,cImgIn,predictedCentroidsList,maxAssign):
        area = np.empty(0)
        centroids = np.empty((2,0))
        print("centroids start: "+str(centroids))
        isFirst = True
        # make this a percentage of the frame
        cImg = cImgIn.copy()
        cImg[0:int(np.ceil(cImg.shape[0]*self.percentFrameRemoveY[0])),:] = 0
        cImg[-int(np.ceil(cImg.shape[0]*self.percentFrameRemoveY[1])):-1,:] = 0

        cImg[:,1:int(np.ceil(cImg.shape[1]*self.percentFrameRemoveX[0]))] = 0
        cImg[:,-int(np.ceil(cImg.shape[1]*self.percentFrameRemoveX[1])):-1] = 0
        
        bottomEdge = int(cImg.shape[0]*(1-self.percentFrameRemoveY[1])) - 3
        topEdge = int(cImg.shape[0]*self.percentFrameRemoveY[0]) + 3        

        #print(topEdge)
        #print(bottomEdge)

        # MAKE side view
        sideView = cImgIn.copy();
        sideView[:,1:int(np.ceil(cImg.shape[1]*(self.percentFrameRemoveX[1] + .25)))] = 0
        sideView[0:int(np.ceil(cImg.shape[0]*self.percentFrameRemoveY[0])),:] = 0
        sideView[-int(np.ceil(cImg.shape[0]*self.percentFrameRemoveY[1])):-1,:] = 0
        sideView = 255*sideView.copy().astype('uint8')
        
        if np.any(cImg):
            tmpcImg = 255*cImg.copy().astype('uint8')
            tmpGarbage,contoursAll,hierarchy = cv2.findContours(tmpcImg,cv2.RETR_TREE ,1) # changed from ...(tmpcImg,2,1)
            contours = [contoursAll[i] for i in range(len(contoursAll)) if hierarchy[:,i,-1] == -1]
            labelVec  = np.arange(0,len(contours))

            ####################
            matchedCentroids = [ [] for i in range(len(contours)) ]
            predCentroidIsUsed = np.zeros(len(contours),dtype = bool)
            #startEval = time.time()
            #print(len(contours))
            for iObj in labelVec:
                cCont = contours[iObj]
                
                cHull = cv2.convexHull(cCont,returnPoints = False)
                defects = cv2.convexityDefects(cCont,cHull)
                if defects is not None:
                    contYArray = np.squeeze(np.array(cCont[cHull]))[:,1]
                    if np.any(contYArray >= bottomEdge) or np.any(contYArray <= topEdge): #any([(cPnt[0][1] >= bottomEdge) or (cPnt[0][1] <= topEdge)   for cPnt in cCont]): #cCont: # check to see is contour is overlapping with the edges
                        #print('on edge')
                        #print('time to check: ' + format(time.time() - startEval,'0.06'))
                        nDiffs = 1
                    else:
                        caveKeep = self.concavityThresh*256 < defects[:,:,3] # multiply by the bit size (256)
                        nDiffs = int(np.ceil(sum(caveKeep)[0]/2.0) + 1)
                        #if nDiffs > 1:
                        #    print('concavity: ' + str(nDiffs))
                else:
                        nDiffs = 1
                
                rows,cols = cImg.shape[:2]
                
                [vx,vy,x,y] = cv2.fitLine(cCont, cv2.DIST_L2,0,0.01,0.01)
                
                cArea = cv2.contourArea(cCont)                

                if cArea > self.minBlobArea:
                    #print('extra')
                    #find the max and min of x and y
                    minx = np.min(cCont[:,:,0])
                    maxx = np.max(cCont[:,:,0])
                    miny = np.min(cCont[:,:,1])
                    maxy = np.max(cCont[:,:,1])
                    
                    print("x: ({:.1f},{:.1f}). y: ({:.1f},{:.1f})".format(minx,maxx,miny,maxy))
                    
                    for pred in predictedCentroidsList:
                        #check if this predicted centroid is inside this area
                        if minx < pred[0] < maxx and miny < pred[1] < maxy and not predCentroidIsUsed[iObj]:
                            print("in contour")
                            matchedCentroids[iObj].append(np.array([pred]))
                            predCentroidIsUsed[iObj] = True
                        #check if this predicted centroid is just outside this area
                        elif minx - maxAssign < pred[0] < maxx + maxAssign and miny - maxAssign < pred[1] < maxy + maxAssign and not predCentroidIsUsed[iObj]:
                            print("near contour")
                            matchedCentroids[iObj].append(np.array([[(pred[0]+x)/2,(pred[1]+y)/2]]))
                            #print("matched: "+str(len(matchedCentroids))+" "+str(matchedCentroids))
                            predCentroidIsUsed[iObj] = True
                    if not matchedCentroids[iObj]==None and len(matchedCentroids[iObj]) > 1:
                        print("pred centroids")
                        #add the predicted contours as actual contours to final centroids list
                        print("centroids :"+str(centroids))
                        for i in range(len(matchedCentroids[iObj])):
                            print("matches: "+str(matchedCentroids[iObj][i][0]))
                            if isFirst:
                                isFirst = False
                                centroids = np.array(matchedCentroids[iObj][i][0])
                            else:
                                centroids = np.column_stack((centroids,matchedCentroids[iObj][i][0]))
                    else:
                        #add actual centroid
                        print("actual centroid")
                        if isFirst:
                            isFirst = False
                            centroids = np.array([x,y])
                        else:
                            centroids = np.column_stack((centroids,np.array([x,y])))
                ####################
                    if nDiffs == 1 and self.frameCount > 35:
                        '''
                        if cArea > 2.1*self.maxBlobSize:
                            centroids = np.column_stack((centroids,np.array([x,y])))
                            centroids = np.column_stack((centroids,np.array([x,y]))) #so 3 pills-- 3 centroids total
                            
                            nDiffs = 3
                            area = np.append(area,cArea/3)

                            cBlobSplit = cArea/self.maxBlobSize
                            self.blobSplit = np.max([self.blobSplit,cBlobSplit])
                            print('{} double blob split {}'.format(datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S'), cBlobSplit))
                        '''
                        if cArea > 1.25*self.maxBlobSize and len(matchedCentroids[iObj]) == 1:
                            
                            area = np.append(area,cArea/2)
                            centroids = np.column_stack((centroids,np.array([x,y])))
                            nDiffs = 2
                            cBlobSplit = cArea/self.maxBlobSize
                            self.blobSplit = np.max([self.blobSplit,cBlobSplit])
                            print('{} blob split {}'.format(datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S'), cBlobSplit))
                    elif nDiffs > 1:
                        for count in range(nDiffs - 1):
                            area = np.append(area,cArea/nDiffs)
                            centroids = np.column_stack((centroids,np.array([x,y])))                   

                    area = np.append(area,cArea/nDiffs)                   

            
            if self.sideViewSizeAdjust != 0 and self.frameCount <= 5:
                # add in sideview areas
                side_areas = np.array(self.getSideAreas(sideView))
                area = np.append(area, side_areas)
           
            #print('time to check:  {0:.06f}'.format(time.time() - startEval))
            if area.size > 0:
                if (self.frameCount<=500 and area.size > 0): # if we've seen 50 (500 frames of pills) pills, really no need to continue computation
                    self.maxBlobSize = np.max(np.append(area,self.maxBlobSize)) # take the max
                    self.areaVec[self.frameCount - 1] = np.max(area) # max of the ones in this frame
                    self.maxBlobSize = np.percentile(self.areaVec[0:self.frameCount], 98) # 98th percentile (reduce some outliers)                   
                    #self.minAreaVec[self.frameCount - 1] = np.min(area) # min of the ones in this frame
                    #self.minBlobArea = np.max((self.minBlobAreaAbs,
                    #                    np.percentile(self.minAreaVec[0:self.frameCount], 2))) # 98th percentile (reduce some outliers)
                    self.minBlobArea = np.max(((self.minBlobAreaPercentage/100.0)*self.maxBlobSize, self.minBlobAreaAbs))
                    self.frameCount = self.frameCount + 1 # increase framecount

                if (self.frameCount <= 50): # 50 frames of pills to check clear est
                    self.clearEst = self.clearEst + int((len(hierarchy[0]) - len(contours)) > 1) # create an estimate for how clear it is
                
        centroids = np.transpose(centroids)        
        #print("centroids: "+str(centroids))
        return area, centroids

        
    def getSideAreas(self, sideView):
        tmpGarbage,contoursAll,hierarchy = cv2.findContours(sideView,cv2.RETR_TREE ,1) # changed from ...(tmpcImg,2,1)
        contours = [contoursAll[i] for i in range(len(contoursAll)) if hierarchy[:,i,-1] == -1]
        side_areas = []
        for cCont in contours:
            cArea = cv2.contourArea(cCont)
            if cArea > self.minBlobArea:
                side_areas.append(cArea*self.sideViewSizeAdjust/100.0)
        return side_areas
