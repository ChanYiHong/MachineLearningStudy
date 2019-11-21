#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Make Transition Matrix
TM = {'study':[0.60, 0.05, 0.05, 0.20], 'rest' : [0.15, 0.80, 0.15, 0.15], 'walk' : [0.05, 0.10, 0.50, 0.15], 'eat' : [0.20, 0.05, 0.30, 0.50]}
TM = pd.DataFrame(TM, columns = ['study', 'rest', 'walk', 'eat'])
TM = TM.rename(index = {0 : 'study', 1 : 'rest', 2 : 'walk', 3 : 'eat'})
TM


# In[2]:


# Calculate the probability for the sequence of observations below.
observation = ['rest', 'eat', 'study', 'study', 'walk', 'rest']
prob = []
for i in range(len(observation)-1):
    curr = observation[i]
    fut = observation[i+1]
    
    prob.append(TM.loc[curr,fut])
    
prob


# In[3]:


# Result
np.product(prob)

