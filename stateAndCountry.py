#!/usr/bin/env python
# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

# NaN인지 아닌지 판별 함수. NaN끼리 비교는 항상 False. 그 점을 이용
def isNaN(num):
    return num != num

## 일단 위도 경도 쓰레기값 찾아서 지움 (1개밖에 없긴함)
#맨 처음 수정되지 않은 초기 데이터 셋
originalData = pd.read_csv('/Users/hcy/Desktop/수업자료/3학년 1학기/데이터과학/데과 텀프/ufo-sightings/scrubbed.csv',encoding='utf-8')

#초기 데이터 셋의 행과 열의 갯수가 몇개인지?
print("초기의 행, 열 개수")
print(originalData.shape)

latitude = originalData['latitude'].to_numpy()
longitude = originalData['longitude '].to_numpy()

#latitude dirty data 검색
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

for i in range(len(dirty_index)):
    originalData = originalData.drop(dirty_index[i])

print("총 한개 지워짐")
print(originalData.shape)
originalData = originalData.reset_index(drop=True)

#longitude dirty data 검색
dirty_data = []
dirty_index = []
for i in range(len(longitude)):
    try:
        longitude[i] = float(longitude[i])
    except ValueError:
        dirty_index.append(i)
        dirty_data.append(longitude[i])

print("longitude는 dirty data가 없음")
print(dirty_data)
print(dirty_index)


print("------------------------------------------")
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

print("-----------")


# 음수 계산 문제를 해결하기 위해 list로 바꿨음.

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
    
        
print("4번 이후")
print(originalData.iloc[:,2:4].head(100))

# -----------------------------------------------------------------
# -----------------------------------------------------------------

# State 기준으로 country 채우기
# 5번
# 위에서 수정되서 내려온 데이터들임
state = originalData['state'].to_numpy()
country = originalData['country'].to_numpy()

# country 빈거 검색
dirty_index = []
for i in range(len(originalData)):
    if isNaN(country[i]) == True:
        dirty_index.append(i)
        
#print(dirty_index)

print("여기까진?")


# State 검색, 똑같은거 있으면 채움
for i in range(len(dirty_index)):
    for j in range(len(originalData)):
        if originalData.iloc[j,2] == originalData.iloc[dirty_index[i],2]:
            originalData.iloc[dirty_index[i],3] = originalData.iloc[j,3]
            break
        # 혹시 똑같은게 없으면 걍 us로 채움
        if j == len(originalData)-1:
            originalData.iloc[dirty_index[i],3] = 'us'
    
print("5번 이후")            
print(originalData.iloc[:,2:4].head(100))      
print("똑똑")
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
    
from collections import Counter

# 국가별 state들의 mode 계산 (최빈값)
data = Counter(us_state)
# data.most_common(1) 하면 [('tx',1000)] 이런식으로 들어감. list안의 tuple
us_mode = data.most_common(1)[0][0]

data = Counter(gb_state)
gb_mode = data.most_common(1)[0][0]

data = Counter(ca_state)
ca_mode = data.most_common(1)[0][0]

data = Counter(au_state)
au_mode = data.most_common(1)[0][0]

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


# In[ ]:




