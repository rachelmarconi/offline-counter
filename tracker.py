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

def tracker(vDetected,FPS,maxAssign):
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
        
        centroids, mask = detectObjects(bA,frame)
        tracks = predictNewLocationsOfTracks(tracks)
        assignments, unassignedTracks, unassignedDetections = detectionToTrackAssignment(tracks,centroids,maxAssign)
        
        tracks = updateAssignedTracks(tracks,centroids,assignments)
        tracks = updateUnassignedTracks(tracks,unassignedTracks)
        tracks = deleteLostTracks(tracks)
        tracks,centroids,countVec,nextId = createNewTracks(tracks, centroids, unassignedDetections,countVec,iFrame,nextId,estVelStart)
        
        frameTime[iFrame] = time.clock() - startTime
    return countVec, frameTime

def predictNewLocationsOfTracks(tracks,numFrames = 1,leftEdge= 140,frameWidth = 300):
    predictedCentroidsList = np.zeros((len(tracks),2))
    for i in range(len(tracks)):
        # Predict the current location of the track.
        predictedCentroid = tracks[i]['predictor'].predict(numFrames)
        predictedCentroidsList[i] = predictedCentroid[0] #only x,y
        '''
        if predictedCentroid[0][0] > frameWidth: #past the left wall 
            predictedCentroid[0][0] = frameWidth #set new location to allong the wall
            tracks[i]['predictor'].estVelocity[0][0] = 0 #no further left velocity

        if predictedCentroid[0][0] < leftEdge:
            predictedCentroid[0][0] = leftEdge
            tracks[i]['predictor'].estVelocity[0][0] = 0
        '''
        if predictedCentroid[0][1] > 350: #arbitrary very large y position
            tracks[i]['consecutiveInvisibleCount'] = 1000 #arbitrary very large invisible--get rid of it
        else:
            print("    {}\tid: {}".format(predictedCentroid,tracks[i]['id']))
        # Shift the bounding box so that its center is at.
        tracks[i]['center'] = predictedCentroid
        
    return tracks,predictedCentroidsList

def detectObjects(bA,frame,predictedCentroidsList,maxAssign):
    # Detect foreground.
    mask = frame.astype('int16')
    
    #mask = imfill(mask);
    mask = cv2.GaussianBlur(mask,(3,3),0)
    mask = mask.astype('bool')
    
    # Perform blob analysis to find connected components.
    area,centroids = bA.step(mask,predictedCentroidsList,maxAssign)
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

def createNewTracks(tracks, centroids, unassignedDetections,countVec,iFrame,estVelStart):    
    centroidsNew = centroids[unassignedDetections, :];
    
    for i in range(centroidsNew.shape[0]):
        centroid = centroidsNew[[i],:]
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

def step(frame,bA,tracks,maxAssign,curId,estVelStart,numFrames):
    print("step")
    tracksPred,predictedCentroidsList = predictNewLocationsOfTracks(tracks,numFrames,bA.leftEdge,frame.shape[1])
    area,centroids, mask = detectObjects(bA,frame,predictedCentroidsList,maxAssign)
    
    assignments, unassignedTracks, unassignedDetections = detectionToTrackAssignment(tracksPred,centroids,maxAssign)
    
    tracksAssign = updateAssignedTracks(tracksPred,centroids,assignments)
    tracksUnassign = updateUnassignedTracks(tracksAssign,unassignedTracks)
    
    tracksUnassign,nextId = updateTrackCountId(tracksUnassign,curId)
    
    tracksDelete = deleteLostTracks(tracksUnassign)
    tracksNew,centroids,nada = createNewTracks(tracksDelete, centroids, unassignedDetections,None,None,estVelStart)
    
    return tracksNew,nextId