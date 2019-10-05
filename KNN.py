#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np 
import pandas as pd 
import matplotlib as plt

#새로운 값이 추가되면 기존의 training set의 값들과 거리를 구한다.
def calculateDistance(height, weight):
    distance = []
    for i in range(len(x)):
        tempHeight = pow((x[i:i+1,0:1] - height),2)
        tempWeight = pow((x[i:i+1,1:2] - weight),2)
        tempHeight = np.float64(tempHeight)
        tempWeight = np.float64(tempWeight)
        tempDistance = np.sqrt(tempHeight + tempWeight)
        distance.append(tempDistance)
        
    distanceSeries = pd.Series(distance)
    distanceSeries = distanceSeries.sort_values()
    return distanceSeries

#Size값을 분류
def classificationSize(k, seri):
    kRange = seri.head(k)
    index = kRange.index
    index = index.to_numpy()
    size = []
    for i in index:
        temp = data[i,2:3]
        temp.tolist()
        size.append(temp)
    
    size = pd.Series(size)
    return size.mode()
    
url = "/Users/hcy/Desktop/수업자료/3학년 1학기/데이터과학/데과 과제/ClassficationAlgorithms/T-shirtSize.xlsx"

dataset = pd.read_excel(url)
data = dataset.to_numpy()

x = data[:,:-1]

resultSeries = calculateDistance(161,61)
resultSize = classificationSize(5, resultSeries)
print("result :", resultSize.to_numpy())


# In[ ]:




