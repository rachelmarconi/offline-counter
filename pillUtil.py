# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:49:30 2018

@author: Joe
"""
import numpy as np
#import serial
#import serial.tools.list_ports
import time
#from initPillUtil import initTrackObj
import tracker as tr

def assignTracks(costMatrix,tooFar):
    if costMatrix.shape[0] == 0:
        assignments = np.zeros((0,2))
        unassignedTracks = np.zeros((0,1))
        unassignedDetections = np.arange(costMatrix.shape[1])
    else:
        absCostMatrix = np.abs(costMatrix)
        absCostMatrix[costMatrix < 0] = 1.3*absCostMatrix[costMatrix < 0]
        
        assignments = np.zeros((0,2))
        unassignedDetections = np.arange(costMatrix.shape[1])
        
        orderOfDetections = absCostMatrix.min(0).argsort()
        
        for i in orderOfDetections:
            minValue = absCostMatrix[:,i].min()
            minInd = absCostMatrix[:,i].argmin()
            if (minValue < tooFar):
                assignments = np.row_stack((assignments,np.array([[minInd,i]])))
                absCostMatrix[minInd,:] = np.inf
                unassignedDetections = unassignedDetections[i != unassignedDetections]
        
        unassignedTracks = np.setdiff1d(list(range(costMatrix.shape[0])),assignments[:,0])
        #unassignedTracks = np.arange(costMatrix.shape[1])
        #unassignedTracks = unassignedTracks.difference(assignments[:,1])
        
        assignments = assignments[assignments[:,0].argsort(axis = 0),]
    return assignments, unassignedTracks, unassignedDetections

class myPillPredictorSimple(object):
    def __init__(self):
        # state and state description
        self.centroid = np.array([[0,0]],'double')
        self.centroidPrevious = np.array([[0,0]],'double')
        self.estVelocity = np.array([[0,0]],'double')
        
        self.minVertVel = 0
        

    def predict(self,numFrames = 1):
        centroidPredict = self.centroid + (self.estVelocity*numFrames)
        self.centroidPrevious = self.centroid.copy()
        self.centroid = centroidPredict.copy()
        
        return centroidPredict.copy()

    def correct(self,centroidMeasure):
        self.estVelocity = np.mean(np.stack((centroidMeasure - self.centroidPrevious,self.estVelocity)),axis = 0)
        
        if self.estVelocity[:,1] < self.minVertVel:
            self.estVelocity[:,1] = self.minVertVel # force the j velocity to be at least 45:
        
        self.centroid = centroidMeasure.copy()
        return centroidMeasure
    
    def distance(self,centroidList):
        diffList = self.centroid - centroidList
        centroidDistance = np.sign(diffList[:,1])*np.sqrt(np.sum(diffList**2,axis = 1))
        
        return centroidDistance
    
def preProcFrame(cFrame,backgroundModel):
    #frame = np.flip(abs(backgroundModel - cFrame).astype('uint8'),0)
    #frame = (backgroundModel - cFrame).astype('uint8')
    #frame = np.flip((frame - cFrame).astype('uint8'),1)
    frame = np.flip(np.flip(abs(backgroundModel - cFrame).astype('uint8'),0),1)
    frame[frame < 100] = 0
    frame[frame >= 55] = 1
    return frame

def bytes2int(cBytes):
    result = 0
    for b in cBytes:
        result = result * 256 + int(b)
    
    return result

def frameTime2Num(timeFromLast):
    if timeFromLast > .007:
        numFrames = np.round(timeFromLast/0.002083)/2.0
    else:
        numFrames = 1
    return numFrames

def fileEncode(key, string):
    encoded_chars = []
    for i in range(len(string)):
        key_c = key[i % len(key)]
        encoded_c = chr(ord(string[i]) + ord(key_c) % 256)
        encoded_chars.append(encoded_c)
    encoded_string = "".join(encoded_chars)
    return encoded_string

def fileDecode(key, string):
    encoded_chars = []
    for i in range(len(string)):
        key_c = key[i % len(key)]
        encoded_c = chr(ord(string[i]) - ord(key_c) % 256)
        encoded_chars.append(encoded_c)
    decoded_string = "".join(encoded_chars)
    return decoded_string
