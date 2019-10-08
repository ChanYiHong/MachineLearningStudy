#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import math
import warnings
from matplotlib import pyplot as plt
warnings.filterwarnings(action='ignore')

# 맨 처음 수정되지 않은 초기 데이터 셋
## 일단 위도 경도 쓰레기값 찾아서 지움 (1개밖에 없긴함)
originalData = pd.read_csv('/Users/hcy/Desktop/수업자료/3학년 1학기/데이터과학/데과 텀프/ufo-sightings/scrubbed.csv',encoding='utf-8')

# NaN인지 아닌지 판별 함수. NaN끼리 비교는 항상 False. 그 점을 이용
def isNaN(num):
    return num != num

# 초기 데이터 셋의 행과 열의 갯수가 몇개인지?
print("초기의 행, 열 개수")
print(originalData.shape)

latitude = originalData['latitude'].to_numpy()
longitude = originalData['longitude '].to_numpy()
duration = originalData['duration (seconds)'].to_numpy()

# latitude dirty data 검색
print("cleaning latitude dirty data")
dirty_data = []
dirty_index = []
for i in range(len(latitude)):
    try:
        latitude[i] = float(latitude[i])
    except ValueError:
        dirty_index.append(i)
        dirty_data.append(latitude[i])

print(dirty_data)
print(dirty_index)

originalData = originalData.drop(dirty_index)


print("총 한개 지워짐")
print(originalData.shape)

longitude = originalData['longitude '].to_numpy()
# longitude dirty data 검색
print("cleaning longitude dirty data")
dirty_data = []
dirty_index = []
for i in range(len(longitude)):
    try:
        longitude[i] = float(longitude[i])
    except ValueError:
        dirty_index.append(i)
        dirty_data.append(longitude[i])

print("longitude는 dirty data가 없음")
print(originalData.shape)

print("------------------------------------------")

# Duration dirty data 검색
print("cleaning duration dirty data")
dirty_data = []
dirty_index = []
for i in range(len(duration)):
    try:
        duration[i] = float(duration[i])
    except ValueError:
        dirty_index.append(i)
        dirty_data.append(duration[i])

print(dirty_data)
print(dirty_index)


# for j in range(len(dirty_index)):
#     originalData = originalData.drop(dirty_index[j])
#
# originalData = originalData.reset_index(drop=True)
originalData = originalData.drop(dirty_index)

print("총 세개 지워짐")
print(originalData.shape)

print("------------------------------------------")

# Duration scatter plot
print("duration scatter plot")
dirty_data = []
dirty_index = []
cleaned_duration = originalData['duration (seconds)'].to_numpy()
for i in range(len(cleaned_duration)):
    try:
        cleaned_duration[i] = float(cleaned_duration[i])
    except ValueError:
        dirty_index.append(i)
        dirty_data.append(cleaned_duration[i])

y = [0 for _ in range(len(cleaned_duration))]
print(len(y))
plt.figure()
plt.scatter(cleaned_duration, y)
plt.show()

# cleaning outliers
outlier = 0
for i in range(len(originalData)):
    if cleaned_duration[i] > 20000000:
        originalData = originalData.drop(i)
        outlier = outlier + 1

print(originalData.shape)
print(outlier)


print("--------------------------------------------------------")
# 4번
# Country가 비어있으면 위도와 경도를 통해 country를 채운다.

print("4번 이전")
print(originalData.iloc[:,2:4].head(100))

state = originalData['state'].to_numpy()
country = originalData['country'].to_numpy()
latitude = originalData['latitude'].to_numpy()
longitude = originalData['longitude '].to_numpy()

# state와 country 둘다 빈거 검색
dirty_index = []
for i in range(len(originalData)):
    if isNaN(country[i]) == True:
        dirty_index.append(i)
        
#print(dirty_index)

print("---------------------------------------------------------")


# 음수 계산 문제를 해결하기 위해 list로 바꿨음.

from collections import Counter
data = Counter(originalData.iloc[:,3])
print(data)
longitude = longitude.tolist()
latitude = latitude.tolist()

# 음수 계산문제의 원흉은 latitude였음. 혼자 string임 ㅅㅂ
for i in range(len(latitude)):
    latitude[i] = float(latitude[i])

print(type(latitude[0]))
print(type(longitude[0]))


# 미국(us)   latitude : 24 ~ 48, longitude : -67 ~ -125
# 영국(gb)   latitude : 49 ~ 52, longitude : -2 ~ -8
# 캐나다(ca) latitude : 49 ~ 82 longitude : -62 ~ -140
# 호주(au)   latitude : -11 ~ -38 longitude : 113 ~ 152
# 독일(de)은 무시하고 그 외 미국으로 넣어버림

for i in range(len(dirty_index)):
    if latitude[dirty_index[i]] >= 24.0 and latitude[dirty_index[i]] <= 48.0:
        if longitude[dirty_index[i]] >= (-125.0) and longitude[dirty_index[i]] <= (-67.0):
            originalData.iloc[dirty_index[i],3] = 'us'
            continue;
            
    if latitude[dirty_index[i]] >= 49.0 and latitude[dirty_index[i]] <= 52.0:
        if longitude[dirty_index[i]] >= (-8.0) and longitude[dirty_index[i]] <= (-2.0):
            originalData.iloc[dirty_index[i],3] = 'gb'
            continue;
            
    if latitude[dirty_index[i]] >= 49.0 and latitude[dirty_index[i]] <= 82.0:
        if longitude[dirty_index[i]] >= (-140.0) and longitude[dirty_index[i]] <= (-62.0):
            originalData.iloc[dirty_index[i],3] = 'ca'
            continue;
            
    if latitude[dirty_index[i]] >= (-38.0) and latitude[dirty_index[i]] <= (-11.0):
        if longitude[dirty_index[i]] >= 113.0 and longitude[dirty_index[i]] <= 152.0:
            originalData.iloc[dirty_index[i],3] = 'au'
            continue;
            
    originalData.iloc[dirty_index[i],3] = 'us'

from collections import Counter
data = Counter(originalData.iloc[:,3])
print(data)
        
print("4번 이후")
print(originalData.iloc[:,2:4].head(100))

# -----------------------------------------------------------------
# -----------------------------------------------------------------


# 비어있는 state 를 각 country의 mode값으로 채우기
# 6번
# 각 country의 mode 계산

# 위에서 수정되서 내려온 데이터들임
state = originalData['state'].to_numpy()
country = originalData['country'].to_numpy()

us_index = []
gb_index = []
ca_index = []
au_index = []
print("여기까지 오나요?")

# 각각 국가들 index 모음
for i in range(len(originalData)):
    if country[i] == 'us':
        us_index.append(i)
    elif country[i] == 'gb':
        gb_index.append(i)
    elif country[i] == 'ca':
        ca_index.append(i)
    elif country[i] == 'au':
        au_index.append(i)
    else:
        us_index.append(i)
        
#print(us_index)
#print(gb_index)
#print(ca_index)
#print(au_index)

us_state = []
gb_state = []
ca_state = []
au_state = []

# 국가들 index를 통해서 해당 국가의 state를 저장
for i in range(len(us_index)):
    us_state.append(state[us_index[i]])

for i in range(len(gb_index)):
    gb_state.append(state[gb_index[i]])
    
for i in range(len(ca_index)):
    ca_state.append(state[ca_index[i]])
    
for i in range(len(au_index)):
    au_state.append(state[au_index[i]])
    
#print(us_state)
    
from collections import Counter

# 국가별 state들의 mode 계산 (최빈값)
data = Counter(us_state)
print(data)
# data.most_common(1) 하면 [('tx',1000)] 이런식으로 들어감. list안의 tuple
us_mode = data.most_common(1)[0][0]
print(us_mode)

data = Counter(gb_state)
print(data)

gb_mode = data.most_common(1)[0][0]
print(gb_mode)

data = Counter(ca_state)
print(data)
ca_mode = data.most_common(1)[0][0]
print(ca_mode)

data = Counter(au_state)
print(data)
au_mode = data.most_common(1)[0][0]
print(au_mode)

# 최빈값을 통해 비어있는 state를 채움
for i in range(len(originalData)):
    if isNaN(state[i]) == True and country[i] == 'us' :
        originalData.iloc[i,2] = us_mode
    elif isNaN(state[i]) == True and country[i] == 'gb' :
        originalData.iloc[i,2] = gb_mode
    elif isNaN(state[i]) == True and country[i] == 'ca' :
        originalData.iloc[i,2] = ca_mode
    elif isNaN(state[i]) == True and country[i] == 'au' :
        originalData.iloc[i,2] = au_mode

        
print("완료")

print(originalData.iloc[:,2:4].head(100))


finishData = originalData

finishData.to_csv("/Users/hcy/Desktop/수업자료/3학년 1학기/데이터과학/데과 텀프/ufo-sightings/sunguk.csv",header=True, index=False)


# In[ ]:




