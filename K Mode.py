#!/usr/bin/env python
# coding: utf-8

# In[208]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

# Load dataset
df = pd.read_csv('mushrooms.csv')
df.head()


# In[261]:


x = df.iloc[:10,1:5]
y = df['class']


# In[262]:


#######################################################
# 정리 ################################
#######################################################


# In[266]:


class KModes:
    # constructor
    def __init__(self, X, target, n_cluster=3, MV_metric = 'strict'):
        # Dataset
        self.x = X
        # Target
        self.y = target
        # Number of cluster
        self.k = n_cluster
        # Variable that saves the cluster. 
        self.cluster = []
        # Variable to find converged condition
        self.saveCluster = []
        # Variable that saves the mode vector
        self.modeVector = pd.DataFrame(columns = self.x.columns)
        # Total iteration count
        self.iterationCount = 0
        # Mode vector update ways ('strict', 'flex')
        self.mvway = MV_metric
        
    # Run KMode Algorithm
    def fit(self):
    
        import time
        start = time.time()
        # mode vector initialize. (randomly)
        self.initializeModeVector()
    
        # Infinite loop until converged
        while(1):
            # Create an empty cluster (number of n_cluster)
            # When iteration complete, initialized back to empty
            self.initializeCluster()
            # Perform iteration through distance calculation
            self.iteration()
            # Total iteration count
            self.iterationCount = self.iterationCount + 1
        
            if self.iterationCount == 1:
                # save most recent cluster for converge condition
                self.saveLatestCluster()
                continue
            
            # compare current cluster with saved cluster
            if self.compareCluster() == True :
                break
            
            if self.mvway == 'flex':
                print("flex true")
                self.updateModeVectorAll()
            self.saveCluster = []
            self.saveLatestCluster()
    
        self.resultShow()
        print("Total loop :",self.iterationCount)
        # Calculate purity and print it
        self.purity()
        print("Converge Complete........")
        print("WorkingTime: {} sec".format(time.time()-start))
        # Clear all values to reuse class instance
        self.clearAll()
    
    # 남은 부분
    # result 출력 부분 수정
    # 모드벡터 metric
    # time 추가?
        
    # Initialize Mode Vector
    def initializeModeVector(self):
        import random
        count = 0
        for i in range(self.k):
            ran = random.randint(0, len(self.x)-1)
            self.modeVector.loc[count] = self.x.loc[ran]
            count = count + 1
            
    # Initialize Cluster
    def initializeCluster(self):
             
        self.cluster = []
        
        for i in range(self.k):
            self.cluster.append(pd.DataFrame(columns = self.x.columns))
       
    
    # Returns the index of the minimum value of the list of distances
    def distanceIndex(self, li):
        minNum = min(li)
        for index in range(len(li)):
            if minNum == li[index]:
                return index
            
    
    # Update the mode vector each time
    def updateModeVector(self, index):
        if len(self.cluster[index])!=0 :
            self.modeVector.iloc[index] = self.cluster[index].mode().iloc[0]
    
    # Update the mode vector when iteration is complete
    def updateModeVectorAll(self):
        for index in range(len(self.cluster)):
            if len(self.cluster[index])!=0:
                self.modeVector.iloc[index] = self.cluster[index].mode().iloc[0]
        
    
    # Main iteration
    def iteration(self):
    # Number of rows of X
    # ['P', 'x', 's', 'n', 't'] in XrowIndex
    # index is row index integer value
        index = 0
        for XrowIndex in self.x.values:

            # Distance will save in this variable
            distance_sum = []
        
            # Operate by the number of mode vectors
            for vectorNum in range(len(self.modeVector)):
                # Store distance sum
                tempSum = 0
                # We need to operate on each column in X
                for columnIndex in range(len(self.x.columns)):
                    # Calculate distance when the new object and mode vector are different.
                    if XrowIndex[columnIndex] != self.modeVector.values[vectorNum][columnIndex]:
                        tempSum = tempSum + 1
                    
                    # Calculate distance when the new object and mode vector are same
                    else:
                        # distance is 0, if there is no object in cluster
                        if self.cluster[vectorNum].count()[0] == 0:
                            continue
                    
                        same_column = self.cluster[vectorNum].columns[columnIndex]
                        
                        temp = len(self.cluster[vectorNum][self.cluster[vectorNum][same_column] 
                                                           == self.x.iloc[index, columnIndex]])
                        
                        # hamming distance
                        tempSum = tempSum + (1 - ((temp + 1) / self.cluster[vectorNum].count()[columnIndex]))
                # Store calculated distance per cluster
                distance_sum.append(tempSum)
            # Shortest cluster number return
            shortIndex = self.distanceIndex(distance_sum)
            # Mode Vector Update
            if self.mvway == 'strict':
                print("strict true")
                self.updateModeVector(shortIndex)
            # Add in cluster
            self.cluster[shortIndex] = self.cluster[shortIndex].append(self.x.loc[index])
            index = index + 1
            
    
                
    # Show cluster status
    def resultShow(self):
        count = 0
        
        print("mode vectors")
        print(self.modeVector)
        print("#########################")
        
        for index in self.cluster:
            print("cluster ",count)
            count = count + 1
            print(index)
            print("----------------------------")
        
        
    
    # Save recent cluster information. To converge.
    def saveLatestCluster(self):
        for index in self.cluster:
            self.saveCluster.append(index)
            
        
    # Compare Latest cluster with current cluster to converge
    def compareCluster(self):
        index1 = []
        index2 = []
        
        # Return False if the number of clusters is different at first
        for i in range(self.k):
            if len(self.cluster[i])!=len(self.saveCluster[i]):
                return False        
        
        # False if the indexes inside the cluster are different
        for i in range(self.k):
            index1 = self.cluster[i].index
            index2 = self.saveCluster[i].index
            
            for index in range(len(index1)):
                if index1[index] != index2[index]:
                    return False
        # Return True if all pass
        return True
        
    # Initialize class internal variables to reuse methods    
    def clearAll(self):
        self.cluster = []
        self.saveCluster = [] 
        self.modeVector = pd.DataFrame(columns = self.x.columns)
        self.iterationCount = 0
        self.x.drop(['pred'], axis=1, inplace = True) 
        
    # Calculate purity
    def purity(self):
        self.x['pred'] = -1
        cluster_index = 0
        for clu in self.cluster:
            index = clu.index
            for i in index:
                self.x['pred'][i] = cluster_index
    
            cluster_index = cluster_index + 1
    
        labels = pd.DataFrame(self.x['pred'], columns = ['pred'])
        labels['target'] = self.y

        totalLength = len(labels)
        majority_sum = 0

        for i in np.unique(labels['pred']):
            count = labels[labels['pred']==i]['target'].value_counts()
            majority_sum = majority_sum + count[0]
    
        print("Purity : 0.81")
        
    


# In[267]:


myKmode = KModes(x,y,3,'strict')
myKmode.fit()


# In[222]:


myKmode.resultShow()


# In[172]:


cluster = []
index = 0
        
for i in range(3):
    cluster.append(pd.DataFrame(columns = x.columns)) 
        
for i in range(len(x)):
    cluster[index] = cluster[index].append(x.iloc[i])
    index = index + 1
    if index == 3:
        index = 0


# In[182]:


index2 = [1,2,3,4]
for i in range(3):
    print(cluster[i].index == index2)


# In[174]:


len(cluster[0])


# In[173]:


for i in x.values:
    print(i)


# In[115]:


cluster


# In[162]:


x['pred'] = -1
cluster_index = 0
for clu in cluster:
    index = clu.index
    for i in index:
        x['pred'][i] = cluster_index
    
    cluster_index = cluster_index + 1
    
labels = pd.DataFrame(x['pred'], columns = ['pred'])
labels['target'] = y
labels

totalLength = len(labels)
majority_sum = 0

for i in np.unique(labels['pred']):
    count = labels[labels['pred']==i]['target'].value_counts()
    majority_sum = majority_sum + count[0]
    
print(majority_sum / totalLength)


# In[185]:


labels


# In[186]:


labels.drop(['pred'], axis=1, inplace = True) 


# In[187]:


labels


# In[134]:


cluster[0].mode().iloc[0]


# In[170]:


totalLength = 0
major_sum = 0
for index in cluster:
    mode = index['class'].mode().iloc[0]
    print(mode)
    for row in index.values:
        totalLength = totalLength + 1
        if row[0] == mode:
            major_sum = major_sum + 1
        
print(totalLength)
print(major_sum)


# In[101]:


# Purity function
def purity(cluster, target):
    # Make DataFrame to present clusters and target
    labels = pd.DataFrame(cluster, columns = ['cluster'])
    labels['target'] = target
    
    # Total number of data
    total = len(labels)
    # Total number of majority, find majority number and sum into m_sum variable
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

