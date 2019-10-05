#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd 
import numpy as np 
import io 
import warnings
warnings.filterwarnings(action='ignore')

from matplotlib import pyplot as plt

#Import the data file
#Distance (meter), Delivery Time (minute)
df = pd.read_csv('/Users/hcy/Desktop/수업자료/3학년 1학기/데이터과학/데과 과제/Lab 4/data_sets_lab4/linear_regression_data.csv',encoding='utf-8')
print(df.head())

#Change data to array
distance = df.iloc[:,0].to_numpy()
deliveryTime = df.iloc[:,1].to_numpy()

#Split data between training dataset, test dataset
trainingDis = distance[:24]
trainingTime = deliveryTime[:24]
testDis = distance[25:]
testTime = deliveryTime[25:]


# In[65]:


#Calculate linear regression
xBar = trainingDis.mean()
yBar = trainingTime.mean()
n = len(trainingDis)
B_numerator = (trainingDis * trainingTime).sum() - n * xBar * yBar
B_denominator = (trainingDis * trainingDis).sum() - n * xBar * xBar
B = B_numerator / B_denominator
A = yBar - B * xBar
print("X = {}\nY = {}".format(trainingDis, trainingTime))
print("Linear regression (y = Bx + A):")
print("B = {}, A = {}".format(B, A))


# In[66]:


#Make Linear regression to use training dataset
plt.scatter(trainingDis,trainingTime)
plt.xlabel('Distance')
plt.ylabel('Delivery Time')
px = np.array([trainingDis.min()-1, trainingTime.max()+1])
py = B * px + A
plt.plot(px, py, color = 'r')
plt.show()


# In[74]:


#Test data
resultTime = []
for i in range(len(testDis)):
    resultTime.append(round(B * testDis[i] + A, 2))

print("Test Result")
print(resultTime)
print("--------------------")
print("Answer value")
print(testTime)


# In[ ]:




