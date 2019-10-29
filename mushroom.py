#!/usr/bin/env python
# coding: utf-8

# In[298]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')


# In[299]:


# Purity function
def purity(cluster, target):
    # Make DataFrame to present clusters and target
    labels = pd.DataFrame(cluster, columns = ['cluster'])
    labels['target'] = target
    
    total = len(labels)
    m_sum = 0
    for i in np.unique(labels['cluster']):
        count = labels[labels['cluster']==i]['target'].value_counts()
        if len(count)==1:
            if count.index[0]==0:
                m_sum = m_sum + count[0]
            else:
                m_sum = m_sum + count[1]
        else:
            if count[0] >= count[1]:
                m_sum = m_sum + count[0]
            else:
                m_sum = m_sum + count[1]

    return m_sum / total    


# In[300]:


# Load dataset
df = pd.read_csv("mushrooms.csv")
df.head()


# In[301]:


# Split dataset into predictor & target
X = df.drop(['class'],axis=1)
y = df['class']
X.head()


# In[302]:


# Print target head
y.head()


# In[303]:


# See if there is null value or not in 'X'
X.isnull().sum()


# In[304]:


# See if there is null value or not in 'y'
y.isnull().sum()


# In[305]:


# Transformation of class values by LabelEncoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for col in X.columns:
    X[col] = encoder.fit_transform(X[col])
    le_y=LabelEncoder()
    y = le_y.fit_transform(y)
X.head()


# In[306]:


# Only use LabelEncoding 
# Parameter : eps = 0.1, min_sample = 5, metric = Euclidean
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.1, min_samples=5, metric='euclidean')
dbscan = dbscan.fit(X)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[307]:


# Only use LabelEncoding 
# Parameter : eps = 0.1, min_sample = 10, metric = Euclidean
dbscan = DBSCAN(eps=0.1, min_samples=10, metric='euclidean')
dbscan = dbscan.fit(X)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[308]:


# Only use LabelEncoding 
# Parameter : eps = 1, min_sample = 5, metric = Euclidean
dbscan = DBSCAN(eps=1, min_samples=5, metric='euclidean')
dbscan = dbscan.fit(X)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[309]:


# Only use LabelEncoding 
# Parameter : eps = 1, min_sample = 10, metric = Euclidean
dbscan = DBSCAN(eps=1, min_samples=10, metric='euclidean')
dbscan = dbscan.fit(X)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[310]:


# Only use LabelEncoding 
# Parameter : eps = 5, min_sample = 5, metric = Euclidean
dbscan = DBSCAN(eps=5, min_samples=5, metric='euclidean')
dbscan = dbscan.fit(X)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[311]:


# Only use LabelEncoding 
# Parameter : eps = 5, min_sample = 20, metric = Euclidean
dbscan = DBSCAN(eps=5, min_samples=20, metric='euclidean')
dbscan = dbscan.fit(X)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[330]:


# Only use LabelEncoding 
# Parameter : eps = 0.1, min_sample = 10, metric = Hamming
dbscan = DBSCAN(eps=0.1, min_samples=10, metric='hamming')
dbscan = dbscan.fit(X)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[331]:


# Only use LabelEncoding 
# Parameter : eps = 1, min_sample = 10, metric = Hamming
dbscan = DBSCAN(eps=1, min_samples=10, metric='hamming')
dbscan = dbscan.fit(X)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[332]:


# Only use LabelEncoding 
# Parameter : eps = 5, min_sample = 10, metric = Hamming
dbscan = DBSCAN(eps=5, min_samples=10, metric='hamming')
dbscan = dbscan.fit(X)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[334]:


# Only use LabelEncoding 
# Parameter : eps = 0.1, min_sample = 10, metric = Manhattan
dbscan = DBSCAN(eps=0.1, min_samples=10, metric='manhattan')
dbscan = dbscan.fit(X)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[335]:


# Only use LabelEncoding 
# Parameter : eps = 1, min_sample = 10, metric = Manhattan
dbscan = DBSCAN(eps=1, min_samples=10, metric='manhattan')
dbscan = dbscan.fit(X)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[336]:


# Only use LabelEncoding 
# Parameter : eps = 5, min_sample = 10, metric = Manhattan
dbscan = DBSCAN(eps=5, min_samples=10, metric='manhattan')
dbscan = dbscan.fit(X)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[312]:


# Use MinMaxScaler to make sure all features are between 0 and 1
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
X_MinMax_train = min_max_scaler.fit_transform(X)
X_MinMax = pd.DataFrame(X_MinMax_train, columns = X.columns)
X_MinMax.head()


# In[313]:


# Use LabelEncoding + MinMaxScaler 
# Parameter : eps = 0.1, min_sample = 10, metric = Euclidean
dbscan = DBSCAN(eps=0.1, min_samples=10, metric='euclidean')
dbscan = dbscan.fit(X_MinMax)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[314]:


# Use LabelEncoding + MinMaxScaler 
# Parameter : eps = 1, min_sample = 10, metric = Euclidean
dbscan = DBSCAN(eps=1, min_samples=10, metric='euclidean')
dbscan = dbscan.fit(X_MinMax)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[316]:


# Use LabelEncoding + MinMaxScaler 
# Parameter : eps = 1, min_sample = 20, metric = Euclidean
dbscan = DBSCAN(eps=1, min_samples=20, metric='euclidean')
dbscan = dbscan.fit(X_MinMax)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[317]:


# Use LabelEncoding + MinMaxScaler 
# Parameter : eps = 5, min_sample = 10, metric = Euclidean
dbscan = DBSCAN(eps=5, min_samples=10, metric='euclidean')
dbscan = dbscan.fit(X_MinMax)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[318]:


# Use LabelEncoding + MinMaxScaler 
# Parameter : eps = 5, min_sample = 20, metric = Euclidean
dbscan = DBSCAN(eps=5, min_samples=20, metric='euclidean')
dbscan = dbscan.fit(X_MinMax)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[337]:


# Use LabelEncoding + MinMaxScaler 
# Parameter : eps = 0.1, min_sample = 10, metric = Hamming
dbscan = DBSCAN(eps=0.1, min_samples=10, metric='hamming')
dbscan = dbscan.fit(X_MinMax)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[338]:


# Use LabelEncoding + MinMaxScaler 
# Parameter : eps = 1, min_sample = 10, metric = Hamming
dbscan = DBSCAN(eps=1, min_samples=10, metric='hamming')
dbscan = dbscan.fit(X_MinMax)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[339]:


# Use LabelEncoding + MinMaxScaler 
# Parameter : eps = 5, min_sample = 10, metric = Hamming
dbscan = DBSCAN(eps=5, min_samples=10, metric='hamming')
dbscan = dbscan.fit(X_MinMax)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[340]:


# Use LabelEncoding + MinMaxScaler 
# Parameter : eps = 0.1, min_sample = 10, metric = Manhattan
dbscan = DBSCAN(eps=0.1, min_samples=10, metric='manhattan')
dbscan = dbscan.fit(X_MinMax)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[341]:


# Use LabelEncoding + MinMaxScaler 
# Parameter : eps = 1, min_sample = 10, metric = Manhattan
dbscan = DBSCAN(eps=1, min_samples=10, metric='manhattan')
dbscan = dbscan.fit(X_MinMax)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[342]:


# Use LabelEncoding + MinMaxScaler 
# Parameter : eps = 5, min_sample = 10, metric = Manhattan
dbscan = DBSCAN(eps=5, min_samples=10, metric='manhattan')
dbscan = dbscan.fit(X_MinMax)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[319]:


# Use StandardScaler convert so that the mean is 0 and the standard deviation is 1
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X_std_train = std_scaler.fit_transform(X)
X_std = pd.DataFrame(X_std_train, columns = X.columns)
X_std.head()


# In[320]:


# Use LabelEncoding + StandardScaler 
# Parameter : eps = 0.1, min_sample = 10, metric = Euclidean
dbscan = DBSCAN(eps=0.1, min_samples=10, metric='euclidean')
dbscan = dbscan.fit(X_std)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[321]:


# Use LabelEncoding + StandardScaler
# Parameter : eps = 1, min_sample = 10, metric = Euclidean
dbscan = DBSCAN(eps=1, min_samples=10, metric='euclidean')
dbscan = dbscan.fit(X_std)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[322]:


# Use LabelEncoding + StandardScaler
# Parameter : eps = 5, min_sample = 10, metric = Euclidean
dbscan = DBSCAN(eps=5, min_samples=10, metric='euclidean')
dbscan = dbscan.fit(X_std)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[324]:


# Use LabelEncoding + StandardScaler
# Parameter : eps = 0.1, min_sample = 10, metric = hamming
dbscan = DBSCAN(eps=0.1, min_samples=10, metric='hamming')
dbscan = dbscan.fit(X_std)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[325]:


# Use LabelEncoding + StandardScaler
# Parameter : eps = 1, min_sample = 10, metric = hamming
dbscan = DBSCAN(eps=1, min_samples=10, metric='hamming')
dbscan = dbscan.fit(X_std)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[326]:


# Use LabelEncoding + StandardScaler
# Parameter : eps = 5, min_sample = 10, metric = hamming
dbscan = DBSCAN(eps=5, min_samples=10, metric='hamming')
dbscan = dbscan.fit(X_std)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[327]:


# Use LabelEncoding + StandardScaler
# Parameter : eps = 0.1, min_sample = 10, metric = Manhattan
dbscan = DBSCAN(eps=0.1, min_samples=10, metric='manhattan')
dbscan = dbscan.fit(X_std)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[328]:


# Use LabelEncoding + StandardScaler
# Parameter : eps = 1, min_sample = 10, metric = Manhattan
dbscan = DBSCAN(eps=1, min_samples=10, metric='manhattan')
dbscan = dbscan.fit(X_std)
labels = dbscan.labels_
result = purity(labels, y)
print(result)


# In[329]:


# Use LabelEncoding + StandardScaler
# Parameter : eps = 5, min_sample = 10, metric = Manhattan
dbscan = DBSCAN(eps=5, min_samples=10, metric='manhattan')
dbscan = dbscan.fit(X_std)
labels = dbscan.labels_
result = purity(labels, y)
print(result)

