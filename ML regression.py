#!/usr/bin/env python
# coding: utf-8

# In[138]:


# The purpose of this code is to look at the difference in performance of each regression.
# (Linear regression, Lidge/Lasso/ElasticNet Regularized regression)
# Use data set for predicting house price
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
warnings.filterwarnings(action='ignore')

# Read dataset from csv file
df = pd.read_csv('housetrain.csv', encoding='utf-8')
df.head()


# In[139]:


# Drop Useless ID column
df.drop(['Id'],axis=1, inplace=True)
df.head()


# In[140]:


# Sum of null values in each attribute
df.isnull().sum()


# In[141]:


# If there any null value is exist, drop them.
df.dropna(axis=1, inplace=True)


# In[142]:


# reduce attribute 80 -> 61
df.head()


# In[143]:


# Use One Hot Encoding
# Increase columns 61 -> 216
df = pd.get_dummies(df)
df.head()


# In[144]:


# Visualize using matplotlib (Just 8 columns)
fig, axs = plt.subplots(figsize = (16,8), ncols = 4, nrows = 2)
lm_features = ['OverallQual','YearBuilt','1stFlrSF','YrSold','BedroomAbvGr','KitchenAbvGr','GarageArea','TotRmsAbvGrd']
for i, feature in enumerate(lm_features):
    row = int(i/4)
    col = i%4
    sn.regplot(x=feature,y='SalePrice',data=df, ax=axs[row][col])


# In[149]:


# Use Simple Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

y_target = df['SalePrice']
X_data = df.drop('SalePrice', axis = 1)

scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)


X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size = 0.2, random_state = 156)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

# Performance is very low... 
print('MAE : {0:.3f} \nMSE : {1:.3f} \nRMSE : {2:.3f} \nR2 Score : {3:.3f}'.format(mae, mse, rmse, r2))


# In[150]:


# Ridge Regressor
# Adjust the alpha value to reduce the high coefficient value (L2 Regularization)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

parameters = {'alpha':[1.0,10.0,100.0,1000.0]}
# Use GridSearch to find best combination of parameters
ridge = Ridge()
grid_ridge = GridSearchCV(ridge, parameters, cv=5)

grid_ridge.fit(X_train, y_train)

y_pred = grid_ridge.predict(X_test)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print('Best parameter:', grid_ridge.best_params_)
print('MAE : {0:.3f} \nMSE : {1:.3f} \nRMSE : {2:.3f} \nR2 Score : {3:.3f}'.format(mae, mse, rmse, r2))

# It seems that overfitting was very severe.
# Performance is much better than before.


# In[151]:


# Lasso Regressor
# Make unnecessary coefficients to zero. (L1 Regularization)
from sklearn.linear_model import Lasso

parameters = {'alpha':[1.0,10.0,100.0,1000.0]}
# Use GridSearch to find best combination of parameters
lasso = Lasso()
grid_lasso = GridSearchCV(lasso, parameters, cv=5)

grid_lasso.fit(X_train, y_train)

y_pred = grid_lasso.predict(X_test)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print('Best parameter:', grid_lasso.best_params_)
print('MAE : {0:.3f} \nMSE : {1:.3f} \nRMSE : {2:.3f} \nR2 Score : {3:.3f}'.format(mae, mse, rmse, r2))

# It seems that overfitting was very severe.
# Performance is much better than before.


# In[152]:


# ElasticNet Regressor
# Use both L1 and L2 regularizations at the same time.
from sklearn.linear_model import ElasticNet

parameters = {'alpha':[1.0,10.0,100.0,1000.0]}
# Use GridSearch to find best combination of parameters
ela = ElasticNet()
grid_ela = GridSearchCV(ela, parameters, cv=5)

grid_ela.fit(X_train, y_train)

y_pred = grid_ela.predict(X_test)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print('Best parameter:', grid_ela.best_params_)
print('MAE : {0:.3f} \nMSE : {1:.3f} \nRMSE : {2:.3f} \nR2 Score : {3:.3f}'.format(mae, mse, rmse, r2))

# It seems that overfitting was very severe.
# Performance is much better than before.

