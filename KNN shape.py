#!/usr/bin/env python
# coding: utf-8

# In[16]:


#import numpy as np
#import pandas as pd

#!/usr/bin/env python
# coding: utf-8

# In[178]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 

from sklearn import neighbors, datasets


warnings.filterwarnings(action='ignore')



#맨 처음 수정되지 않은 초기 데이터 셋
df = pd.read_csv('/Users/hcy/Desktop/수업자료/3학년 1학기/데이터과학/데과 텀프/ufo-sightings/scrubbed.csv',encoding='utf-8')


## 43782 데이터에 latitude가 33q.2008이라는 더티데이터 존재. 오류를 일으켜서 먼저 제거한다. 타 코드에서 주의필요 !!!
print(len(df))
df = df.drop([43782])
print(len(df))

dirty = pd.DataFrame
missing = pd.DataFrame
alldirty = pd.DataFrame
temp = pd.DataFrame

temp = pd.DataFrame
print('all data = ',len(df))

## dirty = other나 unknown인 애들만 모여있는 데이터프레임
dirty = df[df['shape'].isin(['other', 'unknown'])]
print(df['shape'].isnull().sum())
print(len(dirty))

print("number of shape's dirty data + missingdata = ",len(dirty) + 1932)

missing = df.loc[df.isnull()['shape'],:]

print(len(missing))
print(len(dirty))


## alldirty = all of dirty data  in shape  
alldirty = pd.merge(missing,dirty,how='outer')
print(len(alldirty))


temp = df[df['shape'] != 'other']
df = temp[temp['shape']!= 'unknown']

## df중에서 shape가 null값인것 삭제  
df = df.loc[df.notnull()['shape'],:]
print(len(df))

## df    -  shape에서 other,unknown , Null 값 제거한 데이터프레임
## alldirty   -  shape 에서 missing, other, unknown 값만 모아둔 데이터프레임 , 나중에 classification을 통해 채울값들 


# In[179]:


print(df.dtypes)
df['latitude'] = df['latitude'].astype(np.float)


# In[180]:


print(df.dtypes)

X=pd.DataFrame

X = df[['longitude ','latitude']].values
y = df['shape'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

print(len(X_train))
print(len(X_test))

classifier = KNeighborsClassifier(n_neighbors=20)  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test) 

print(y_pred)

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

X = alldirty[['longitude ','latitude']].values
Y = classifier.predict(X)

## Y 는 어레이타입
print(Y)
print(np.unique(Y))



alldirty.iloc[0,4] = Y[0]
print(alldirty.iloc[0,4])

i=0
for i in range(len(alldirty)):
    alldirty.iloc[i,4] = Y[i]
    
print(alldirty.iloc[20,4])
print("done")

df['latitude'] = df['latitude'].astype(object)
df = pd.merge(df,alldirty,how='outer')
print(len(df))
print(df)


# In[ ]:






#df = pd.read_csv('/Users/hcy/Desktop/수업자료/3학년 1학기/데이터과학/데과 텀프/ufo-sightings/scrubbed.csv',encoding='utf-8')

#df.iloc[:,550:560]

#print(df['city'].unique())
#print(df['city'].value_counts())

#print(df['state'].unique())
#print(df['state'].value_counts())

#print(df['country'].unique())
#print(df['country'].value_counts())

#print("")

#print("Amount of 'Country' null value")
#print("Null value : ", df['country'].isnull().sum())
#print("Not Null Value : ", df['country'].notnull().sum())

#print("Sum : ", df['country'].isnull().sum() + df['country'].notnull().sum())

#print("")
#print("Amount of 'State' null value")
#print("Null value : ", df['state'].isnull().sum())
#print("Not Null Value : ", df['state'].notnull().sum())

#print("Sum : ", df['state'].isnull().sum() + df['state'].notnull().sum())

#print("")
#print("Amount of 'Shape' null value")
#print("Null value : ", df['shape'].isnull().sum())
#print("Not Null Value : ", df['shape'].notnull().sum())

#print("Sum : ", df['shape'].isnull().sum() + df['shape'].notnull().sum())

#print("city")
#print(df['city'].isnull().sum())


# In[ ]:




