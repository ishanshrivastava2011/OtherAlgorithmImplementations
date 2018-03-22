#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 19:18:37 2017

@author: ishanshrivastava
"""
import numpy as np
import pandas as pd

def personalizedPageRank(similarityDataFrame,seed,alpha):
    
    for node in similarityDataFrame.columns:
        similarityDataFrame.loc[node,node] = 0
    nodes = list(similarityDataFrame.index)
    df = similarityDataFrame
    
    #Normalizing all the columns so that they sum to 1    
    df = df.div(df.sum(axis=0), axis=1)
    df = df.fillna(value=0)

    for node in df.columns[(df == 0).all()]:
        df.loc[node,node] = 0

    npMatrix = np.matrix(df)
    numNodes = len(npMatrix)
    
    seed = [check(i,nodes) for i in seed]
    seed = list(filter(None.__ne__, seed))
    
    #Creating the Teleportation Vector Vq, where q are the seeds
    Vq = [0]*numNodes
    prob = 1/len(seed)
    for i in range(0,numNodes):
        if i in seed:
            Vq[i]=prob
    Vq = np.array(Vq)  
      
    Uq = Vq.copy()
    Uqi = [0]*numNodes
    maxerr = .000000000001
    while np.sum(np.abs(Uq-Uqi)) > maxerr:
        Uqi = Uq.copy()
        Uq = alpha*(Uq.__matmul__(npMatrix)) + (1-alpha)*Vq
        
    Result = pd.DataFrame(Uq.T)
    return Result

def check(i,nodes):
    if i in nodes:
        return nodes.index(i)
    else:
        return None