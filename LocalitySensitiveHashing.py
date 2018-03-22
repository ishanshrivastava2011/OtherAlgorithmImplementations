#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 19:02:48 2017

@author: ishanshrivastava
"""

from collections import defaultdict
import numpy as np
import random
LHashTables = defaultdict(lambda: defaultdict(list))

def createAHashFunction(d,w):
    r = [random.gauss(0,1) for i in range(d)]
    b = random.uniform(0,w)
    hashFunction = tuple((r,b))
    return hashFunction

#k is the number of hashes per layer
def createAHashFamily(k,d,w):
    hashFunctions = list()
    for kth in range(k):
        hashFunctions.append(createAHashFunction(d,w))
    return hashFunctions

#L is the number of Layers
def createLayers(L,k,d,w):
    layerTables = list()
    for layer in range(L):
        layerTables.append(createAHashFamily(k,d,w))
    return layerTables
        
def getHashKeyForAHashFunction(hashFunction,point,w):
    r = hashFunction[0]
    b = hashFunction[1]
    return str(int((np.dot(r,point.T)+b)/w))

def getHashKeyForAHashFamily(hashFunctions,point,w):
    return "".join([getHashKeyForAHashFunction(hashFunction,point,w) for hashFunction in hashFunctions])

def mapPointIndexToLBuckets(index,L,w,layerTables,inputMatrix):
    global LHashTables
    point = inputMatrix[index]
    for layer in range(L):
        hashFunctions = layerTables[layer]
        key = getHashKeyForAHashFamily(hashFunctions,point,w)
        LHashTables[layer][key].append(index)

def createAndGetLSH_IndexStructure(L,k,d,w,inputMatrix):
    global LHashTables
    layerTables = createLayers(L,k,d,w)        
    for index in range(inputMatrix.shape[0]):
        mapPointIndexToLBuckets(index,L,w,layerTables,inputMatrix)
    return layerTables,LHashTables
