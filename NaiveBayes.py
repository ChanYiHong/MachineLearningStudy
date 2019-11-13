#!/usr/bin/env python
# coding: utf-8

# In[86]:


import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("play_golf.csv")
df.head()


# In[87]:


# Split dataset into predictor & target
X = df.drop('play Golf', axis = 1)
y = df['play Golf']


# In[88]:


# Use LabelEncoder to use naive bayes classifier
from sklearn.preprocessing import LabelEncoder

label = {}
for col in X.columns:
    label[col] = LabelEncoder().fit(X[col])


# In[89]:


label


# In[90]:


# Use LabelEncoder
for col in label.keys():
    X[col] = label[col].transform(X[col])


# In[97]:


X.head()


# In[92]:


# Use sklearn naive bayes classifier
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X,y)


# In[93]:


# Predict instance
# Outlook = sunny, Temperature = cool, Humidity = high, Wind = false
new = {'Outlook' : ['sunny'], 'Temp' : ['cool'], 'Humidity' : ['high'], 'Wind' : False}
new = pd.DataFrame(new, columns = ['Outlook','Temp','Humidity','Wind'])
new


# In[94]:


# LabelEncoder for new data
for col in label.keys():
    new[col] = label[col].transform(new[col])
    
new


# In[95]:


# Predict new instance
pred = gnb.predict(new)


# In[96]:


# Result
pred

