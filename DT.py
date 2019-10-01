#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import io
import warnings
import math
warnings.filterwarnings(action='ignore')

df = pd.read_csv('/Users/hcy/Desktop/수업자료/3학년 1학기/데이터과학/데과 과제/Lab 4/data_sets_lab4/decision_tree_data.csv', encoding='utf-8')

global root_entropy, total
total = len(df['interview'])
print(df.head(4))

# Split dataset
trainingSet = df.iloc[:27,:]
testSet = df.iloc[27:,:]


# In[12]:


DecisionTree = {'depth':['root','child1','child2','child3'],
                'attribute':['','','','']}

Result = {'result1':[], 'result2':[], 'result3':[], 'result[4]':[]}




# In[7]:


def compute_root_entropy(criterion):
    label = df[criterion]
    counts= {}
    counts= label.value_counts()
    
    #compute probability
    probs ={}
    probs = counts/total
    
    root_entropy = 0
    for i in range(len(probs)):
        if(probs[i] != 0):
            root_entropy = root_entropy + (probs[i]*math.log(probs[i],2))
    return abs(root_entropy)


# In[8]:


def compute_child_entropy(criterion, vals, df):
    feature_length = len(feature_cols)
    info_gain = np.zeros((feature_length,1))
    val_length = len(vals)
    
    for index in range(feature_length):
        unique_value = df[feature_cols[index]].unique()
        unique_length = len(unique_value)
        print(unique_value)
    
        prob = np.zeros((unique_length, val_length+1))
        for i in range(val_length):
            for j in range(len(df[criterion])):
                if (df[criterion][j] == vals[i]):
                    if(df[feature_cols[index]][j] == unique_value[0]):
                        prob[0][i] = prob[0][i] + 1
                    elif (df[feature_cols[index]][j]==unique_value[1]):
                        prob[1][i] = prob[1][i] + 1
                    else:
                        prob[2][i] = prob[2][i] + 1
        
        entropy = np.zeros((unique_length,1))
        for j in range(unique_length): #row
            for i in range(val_length): #col
                #compute subtotal
                prob[j][val_length] = prob[j][val_length] + prob[j][i]
            for i in range(val_length):
                prob[j][i] = prob[j][i] / prob[j][val_length]
                if(prob[j][i] != 0):
                    entropy[j] = entropy[j] + (prob[j][i]*math.log(prob[j][i],2))
            
        print(prob)
        
        for j in range(unique_length):
            entropy[j] = abs(entropy[j])
        for j in range(unique_length):
            info_gain[index] = info_gain[index] + (prob[j][val_length]*entropy[j])
        
        info_gain[index] = root_entropy - (info_gain[index]/total)

    info_gain = pd.DataFrame(info_gain)
    info_gain['feature'] = feature_cols
    info_gain.rename(columns = {info_gain.columns[0]: 'information gain'}, inplace=True)
    
    info_gain = info_gain.sort_values(by='information gain', ascending=False)
    info_gain = info_gain.reset_index(drop=True)
    return info_gain


# In[ ]:
print("Training Set")
#Training Set 
root_entropy= compute_root_entropy('interview')
feature_cols = pd.Series(list(df.columns))
feature_cols = feature_cols[:len(feature_cols)-1]

info_gain = compute_child_entropy('interview',[True,False],trainingSet)
DecisionTree['attribute'][0] = info_gain.iloc[0,1]
print(info_gain)
print(DecisionTree)
print()
    
feature_cols = pd.Series(['level','tweets','phd'])
root_entropy = compute_root_entropy('lang')
print(root_entropy)
info_gain = compute_child_entropy('lang', ['java','python','R'],trainingSet)
print(info_gain)
DecisionTree['attribute'][1] = info_gain.iloc[0,1]

print(DecisionTree)
print()

feature_cols = pd.Series(['level','phd'])
root_entropy = compute_root_entropy('tweets')
print(root_entropy)
info_gain = compute_child_entropy('tweets', ['yes','no'],trainingSet)
print(info_gain)
DecisionTree['attribute'][2] = info_gain.iloc[0,1]

print(DecisionTree)
print()

feature_cols = pd.Series(['level'])
root_entropy = compute_root_entropy('phd')
print(root_entropy)
info_gain = compute_child_entropy('phd', ['yes','no'],trainingSet)
print(info_gain)
DecisionTree['attribute'][3] = info_gain.iloc[0,1]

print(DecisionTree)
print()

print("------------------------------------------")
print('Test Set')
print("------------------------------------------")
#Test Set
root_entropy= compute_root_entropy('interview')
feature_cols = pd.Series(list(df.columns))
feature_cols = feature_cols[:len(feature_cols)-1]

info_gain = compute_child_entropy('interview',[True,False],df)
DecisionTree['attribute'][0] = info_gain.iloc[0,1]
print(info_gain)
print(DecisionTree)
print()
    
feature_cols = pd.Series(['level','tweets','phd'])
root_entropy = compute_root_entropy('lang')
print(root_entropy)
info_gain = compute_child_entropy('lang', ['java','python','R'],df)
print(info_gain)
DecisionTree['attribute'][1] = info_gain.iloc[0,1]

print(DecisionTree)
print()

feature_cols = pd.Series(['level','phd'])
root_entropy = compute_root_entropy('tweets')
print(root_entropy)
info_gain = compute_child_entropy('tweets', ['yes','no'],df)
print(info_gain)
DecisionTree['attribute'][2] = info_gain.iloc[0,1]

print(DecisionTree)
print()

feature_cols = pd.Series(['level'])
root_entropy = compute_root_entropy('phd')
print(root_entropy)
info_gain = compute_child_entropy('phd', ['yes','no'],df)
print(info_gain)
DecisionTree['attribute'][3] = info_gain.iloc[0,1]

print(DecisionTree)
print()

# In[ ]:




