# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 18:30:43 2018

@author: joeca
"""
#%%
import numpy as np
import cv2
import time
import sys

from myBlobAnalyzerSide import myBlobAnalyzerSide
import pillUtil

#%%
startTime = 0.0
time_string = ""

def tracker(vDetected,FPS,maxAssign):
    global startTime
    # Create System objects used for reading video, detecting moving objects,
    # and displaying the results.
    bA = myBlobAnalyzerSide()
    
    #### change the frame cuts (this is for the side view included)
    bA.percentFrameRemoveX = [.06,.47]
    bA.percentFrameRemoveY = [.4,.04]
    bA.concavityThresh = 5

    estVelStart = np.array([[0,13]],'double')#15]],'double') alterred for 480fps instead of 400fps  #[0,50].*(120/FPS).*(fliplr(size(frame)./[480,720]));
    tracks = []
    
    nextId = 1 # ID of the next track
    countVec = np.zeros((vDetected.shape[2],1))
    frameTime = np.zeros((vDetected.shape[2],1))
    
    for iFrame in range(vDetected.shape[2]):
        startTime = time.clock()
        frame = vDetected[:,:,iFrame]
        
        centroids, mask = detectObjects(bA, frame)
        tracks = predictNewLocationsOfTracks(tracks)
        assignments, unassignedTracks, unassignedDetections = detectionToTrackAssignment(tracks,centroids,maxAssign)
        
        tracks = updateAssignedTracks(tracks, centroids, assignments)
        tracks = updateUnassignedTracks(tracks, unassignedTracks)
        tracks = deleteLostTracks(tracks)
        tracks, centroids, countVec, nextId = createNewTracks(tracks, centroids, unassignedDetections,
                                                           countVec, iFrame, nextId, estVelStart,
                                                           frame.shape[0] - bA.maxAssign)
        
        frameTime[iFrame] = time.clock() - startTime
    return countVec, frameTime

def predictNewLocationsOfTracks(tracks, numFrames = 1, rightEdge= 140, frameHeight = 300):
    predictedCentroidsList = np.zeros((len(tracks),2))
    for i in range(len(tracks)):
        # Predict the current location of the track.
        predictedCentroid = tracks[i]['predictor'].predict(numFrames)
        #predictedCentroidsList[i] = predictedCentroid[0] #only x,y


        if predictedCentroid[0][0] < 0: #past the left wall 
            predictedCentroid[0][0] = 0 #set new location to along the wall
            tracks[i]['predictor'].estVelocity[0][0] = 0 #no further left velocity

        if predictedCentroid[0][0] > rightEdge:
            predictedCentroid[0][0] = rightEdge
            tracks[i]['predictor'].estVelocity[0][0] = 0
        '''
        if predictedCentroid[0][0] > frameWidth: #past the left wall 
            predictedCentroid[0][0] = frameWidth #set new location to along the wall
            tracks[i]['predictor'].estVelocity[0][0] = 0 #no further left velocity

        if predictedCentroid[0][0] < leftEdge:
            predictedCentroid[0][0] = leftEdge
            tracks[i]['predictor'].estVelocity[0][0] = 0
        '''
        # Shift the bounding box so that its center is at.
        tracks[i]['center'] = predictedCentroid

    tracks = [ tracks[i] for i in range(len(tracks)) if tracks[i]['center'][0][1] <  frameHeight] # 280 frameHeight
    predictedCentroidsList = [ tracks[i]['center'][0] for i in range(len(tracks)) ]

    for i in range(len(tracks)):
        print("    {}\tid: {}".format(tracks[i]['center'], tracks[i]['id']))
        
    return tracks,predictedCentroidsList

def detectObjects(bA,frame,predictedCentroidsList,maxAssign,calibrate,inputMaxBlob,inputMinBlob):
    # Detect foreground.
    mask = frame.astype('int16')
    
    #mask = imfill(mask);
    mask = cv2.GaussianBlur(mask,(3,3),0)
    mask = mask.astype('bool')
    
    # Perform blob analysis to find connected components.
    area,centroids = bA.step(mask,predictedCentroidsList,maxAssign,calibrate,inputMaxBlob,inputMinBlob)
    print("    --------")
    for c in centroids:
        print("    {}".format(c))
    
    return area, centroids, mask

def detectionToTrackAssignment(tracks,centroids,maxAssign):
    nTracks = len(tracks)
    nDetections = centroids.shape[0]
    
    # Compute the cost of assigning each detection to each track.
    cost = np.zeros((nTracks,nDetections),'double')
    for i in range(nTracks):
        cost[i, :] = tracks[i]['predictor'].distance(centroids)
    
    #### resolution correction
    tooFar = maxAssign
    assignments, unassignedTracks, unassignedDetections = pillUtil.assignTracks(cost,tooFar)
    return assignments, unassignedTracks, unassignedDetections

def updateAssignedTracks(tracks,centroids,assignments):
    numAssignedTracks = assignments.shape[0]
    #print(str(numAssignedTracks)+" tracks rn")
    sys.stdout.flush()
    for i in range(numAssignedTracks):
        trackIdx = int(assignments[i, 0])
        detectionIdx = [int(assignments[i, 1])]
        centroid = centroids[detectionIdx, :]
        
        # Correct the estimate of the object's location
        # using the new detection.
        tracks[trackIdx]['predictor'].correct(centroid)
        
        # Replace predicted bounding box with detected
        # bounding box.
        tracks[trackIdx]['center'] = centroid
        
        # Update track's age.
        tracks[trackIdx]['age'] = tracks[trackIdx]['age'] + 1
        
        # Update visibility.
        tracks[trackIdx]['totalVisibleCount'] = tracks[trackIdx]['totalVisibleCount'] + 1
        tracks[trackIdx]['consecutiveInvisibleCount'] = 0
    
    return tracks

def updateUnassignedTracks(tracks,unassignedTracks):
    for i in range(len(unassignedTracks)):
        ind = unassignedTracks[i]
        tracks[ind]['age'] = tracks[ind]['age'] + 1
        tracks[ind]['consecutiveInvisibleCount'] = tracks[ind]['consecutiveInvisibleCount'] + 1
    
    return tracks

def updateTrackCountId(tracks,nextId):
    for i in range(len(tracks)):
        if tracks[i]['id'] == 0:
            if tracks[i]['totalVisibleCount'] > 2:
                tracks[i]['id'] = nextId
                # Increment the next id.
                #print(str(nextId)+" found")
                nextId = nextId + 1
    
    return tracks,nextId

def deleteLostTracks(tracks):
    if len(tracks) < 1:
        return tracks
    
    invisibleForTooLong = 12 #9
    
    # Find the indices of 'lost' tracks.
    lostInds = []
    for i in range(len(tracks)):
        lostInds.append(tracks[i]['consecutiveInvisibleCount'] >= invisibleForTooLong)
    
    # Delete lost tracks.
    tracks  = [tracks[i] for i in range(len(tracks)) if not lostInds[i]]
    
    return tracks

def createNewTracks(tracks, centroids, unassignedDetections,countVec,iFrame,estVelStart, lowestNewTrack=260):
    centroidsNew = centroids[unassignedDetections, :];
    
    for i in range(centroidsNew.shape[0]):
        centroid = centroidsNew[[i],:]
        # MCF: Added a check to prevent adding a new track at the bottom of the frame.
        if centroid[0][1] <  lowestNewTrack:
            predictor = pillUtil.myPillPredictorSimple()

            ### correct for 200FPS and resolution
            predictor.estVelocity = estVelStart
            predictor.minVertVel = predictor.estVelocity[:,1]*.9
            predictor.centroid = centroid
            # Create a new track.
            newTrack = [{'id': 0,'center': centroid,'predictor': predictor, 'age': 1,'totalVisibleCount':1,'consecutiveInvisibleCount':0}]

            # Add it to the array of tracks.
            tracks.extend(newTrack)

            #countVec[iFrame] = countVec[iFrame] + 1 # increment count vec
    
    return tracks,centroidsNew,countVec

def step(frame,bA,tracks,maxAssign,curId,estVelStart,numFrames,calibrate,inputMaxBlob,inputMinBlob):
    global startTime

    startTime = time.time()
    print("step----------------------------------------------")
    tracksPred,predictedCentroidsList = predictNewLocationsOfTracks(tracks,numFrames,bA.rightEdge,frame.shape[0])
    time_string = "ta:{:6.4f}".format((time.time()-startTime) * 1000.)
    bA.time_start = startTime
    bA.time_string = ""
    area,centroids, mask = detectObjects(bA,frame,predictedCentroidsList,maxAssign,calibrate,inputMaxBlob,inputMinBlob)
    time_string += bA.time_string
    time_string += ", tb:{:6.4f}".format((time.time()-startTime) * 1000.)

    assignments, unassignedTracks, unassignedDetections = detectionToTrackAssignment(tracksPred,centroids,maxAssign)
    time_string += ", tc:{:6.4f}".format((time.time()-startTime) * 1000.)
    tracksAssign = updateAssignedTracks(tracksPred,centroids,assignments)
    tracksUnassign = updateUnassignedTracks(tracksAssign,unassignedTracks)
    tracksUnassign,nextId = updateTrackCountId(tracksUnassign,curId)

    tracksDelete = deleteLostTracks(tracksUnassign)
    time_string += ", td:{:6.4f}".format((time.time()-startTime) * 1000.)
    tracksNew,centroids,nada = createNewTracks(tracksDelete, centroids, unassignedDetections,None,None,estVelStart,
                                               frame.shape[0] - bA.maxAssign)
    time_string += ", te:{:6.4f}".format((time.time()-startTime) * 1000.)
    print(time_string)
    return tracksNew,nextId
