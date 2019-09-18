#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
import seaborn as sn
import warnings
warnings.filterwarnings(action="ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

# read in the data using pandas
df = pd.read_csv('car.csv')

df.head()


# In[61]:


# Get Statistics on the Data
df.describe()


# In[62]:


# Get Statistics on the Data
df.info()


# In[63]:


# Check if DataFrame has null data or not
print(df.isna().sum())


# In[64]:


# Label Encoding
# Encoding 'buying' attribute
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(df['buying'])
labels = encoder.transform(df['buying'])
df['buying'] = labels
print('Encoding value:', labels)
print('Encoding class:', encoder.classes_)


# In[65]:


# Label Encoding
# Encoding 'maint' attribute
encoder = LabelEncoder()
encoder.fit(df['maint'])
labels = encoder.transform(df['maint'])
df['maint'] = labels
print('Encoding values:', labels)
print('Encoding class:', encoder.classes_)


# In[66]:


# Label Encoding
# Encoding 'lug_boot' attribute
encoder = LabelEncoder()
encoder.fit(df['lug_boot'])
labels = encoder.transform(df['lug_boot'])
df['lug_boot'] = labels
print('Encoding values:', labels)
print('Encoding class:', encoder.classes_)


# In[67]:


# Label Encoding
# Encoding 'safety' attribute
encoder = LabelEncoder()
encoder.fit(df['safety'])
labels = encoder.transform(df['safety'])
df['safety'] = labels
print('Encoding values:', labels)
print('Encoding class:', encoder.classes_)


# In[68]:


# Replace the more value of a persons attribute with a numerical value
encoder = LabelEncoder()
encoder.fit(df['persons'])
labels = encoder.transform(df['persons'])
df['persons'] = labels
print('Encoding values:', labels)
print('Encoding class:', encoder.classes_)


# In[69]:


encoder = LabelEncoder()
encoder.fit(df['doors'])
labels = encoder.transform(df['doors'])
df['doors'] = labels
print('Encoding values:', labels)
print('Encoding class:', encoder.classes_)


# In[70]:


# Label Encoding
# Encoding 'car' attribute
# acc, good, vgood -> 1
# unacc -> 0
df = df.replace({'unacc':0,'acc':1,'good':1,'vgood':1})


# In[71]:


# Data visualization
df.hist(figsize=(20,20))


# In[72]:


#######################################################
#######################################################
#######################################################


# In[73]:


# Use K-fold Cross Validation (k=10)
from sklearn.model_selection import cross_val_score

# Create a new DecisionTree model
# Use parameter criterion = 'gini'
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(criterion = 'gini', random_state=0)

# Train model with cv of 10
X = df.drop(['car'], axis = 1)
y = df['car']
cv_scores = cross_val_score(dt_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[74]:


# Use K-fold Cross Validation (k=10)
from sklearn.model_selection import cross_val_score

# Create a new DecisionTree model
# Use parameter criterion = 'entropy'
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(criterion = 'entropy', random_state=0)

# Train model with cv of 1
cv_scores = cross_val_score(dt_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[75]:


# Model with the best score average = 0.90462..
# Parameter // criterion = 'gini'
# Make 2D bar chart to display the accuracy score
import matplotlib.pyplot as plt

dt_clf = DecisionTreeClassifier(criterion = 'gini', random_state=0)
cv_scores = cross_val_score(dt_clf, X, y, cv=10)

objects = ('1','2','3','4','5','6','7','8','9','10')
y_pos = np.arange(len(objects))

plt.bar(y_pos, cv_scores, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Score')
plt.title('Accuracy Score')

plt.show()


# In[76]:


# Make Confusion Matrix
# Find the confusion matrix of the model with the best score
# Parameter // criterion = 'entropy'
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

dt_clf = DecisionTreeClassifier(criterion = 'gini', random_state=0)

# Train the model
dt_clf.fit(X_train, y_train)

y_pred = dt_clf.predict(X_test)


# In[77]:


data = pd.DataFrame()
data['y_Actual'] =  y_test
data['y_Predicted'] = y_pred


# In[78]:


confusion_matrix = pd.crosstab(data['y_Actual'],data['y_Predicted'],rownames=['Actual'],colnames=['Predicted'],margins=True)

sn.heatmap(confusion_matrix, annot=True)


# In[79]:


#######################################################
#######################################################
#######################################################


# In[80]:


# Create a new Logistic Regression model
# Use K-fold Cross Validation (k=10)
# Use parameter solver = 'liblinear', max_iter = 50
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(solver = 'liblinear', max_iter = 50)

# Train model with cv of 10
cv_scores = cross_val_score(lr_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[81]:


# Create a new Logistic Regression model
# Use K-fold Cross Validation (k=10)
# Use parameter solver = 'liblinear', max_iter = 100
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(solver = 'liblinear', max_iter = 100)

# Train model with cv of 10
cv_scores = cross_val_score(lr_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[82]:


# Create a new Logistic Regression model
# Use K-fold Cross Validation (k=10)
# Use parameter solver = 'liblinear', max_iter = 200
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(solver = 'liblinear', max_iter = 200)

# Train model with cv of 10
cv_scores = cross_val_score(lr_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[83]:


# Create a new Logistic Regression model
# Use K-fold Cross Validation (k=10)
# Use parameter solver = 'lbfgs', max_iter = 50
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(solver = 'lbfgs', max_iter = 50)

# Train model with cv of 10
cv_scores = cross_val_score(lr_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[84]:


# Create a new Logistic Regression model
# Use K-fold Cross Validation (k=10)
# Use parameter solver = 'lbfgs', max_iter = 100
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(solver = 'lbfgs', max_iter = 100)

# Train model with cv of 10
cv_scores = cross_val_score(lr_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[85]:


# Create a new Logistic Regression model
# Use K-fold Cross Validation (k=10)
# Use parameter solver = 'lbfgs', max_iter = 200
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(solver = 'lbfgs', max_iter = 200)

# Train model with cv of 10
cv_scores = cross_val_score(lr_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[86]:


# Create a new Logistic Regression model
# Use K-fold Cross Validation (k=10)
# Use parameter solver = 'sag', max_iter = 50
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(solver = 'sag', max_iter = 50)

# Train model with cv of 10
cv_scores = cross_val_score(lr_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[87]:


# Create a new Logistic Regression model
# Use K-fold Cross Validation (k=10)
# Use parameter solver = 'sag', max_iter = 100
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(solver = 'sag', max_iter = 100)

# Train model with cv of 10
cv_scores = cross_val_score(lr_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[88]:


# Create a new Logistic Regression model
# Use K-fold Cross Validation (k=10)
# Use parameter solver = 'sag', max_iter = 200
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(solver = 'sag', max_iter = 200)

# Train model with cv of 10
cv_scores = cross_val_score(lr_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[89]:


# Model with the best score average = 0.6881...
# Parameter // solver = 'liblinear', max_iter = 50
# Make 2D bar chart to display the accuracy score
lr_clf = LogisticRegression(solver = 'liblinear', max_iter = 50)
cv_scores = cross_val_score(lr_clf, X, y, cv=10)

objects = ('1','2','3','4','5','6','7','8','9','10')
y_pos = np.arange(len(objects))

plt.bar(y_pos, cv_scores, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Score')
plt.title('Accuracy Score')

plt.show()


# In[90]:


# Make Confusion Matrix
# Find the confusion matrix of the model with the best score
# Parameter // solver = 'lbfgs', max_iter = 50

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

lr_clf = LogisticRegression(solver = 'liblinear', max_iter = 50)

# Train the model
lr_clf.fit(X_train, y_train)

y_pred = lr_clf.predict(X_test)


# In[91]:


data = pd.DataFrame()
data['y_Actual'] =  y_test
data['y_Predicted'] = y_pred


# In[92]:


confusion_matrix = pd.crosstab(data['y_Actual'],data['y_Predicted'],rownames=['Actual'],colnames=['Predicted'],margins=True)

sn.heatmap(confusion_matrix, annot=True)


# In[93]:


#######################################################
#######################################################
#######################################################


# In[94]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = linear, C = 0.1, gamma = 10
from sklearn.svm import SVC
svm_clf = SVC(kernel='linear', C = 0.1, gamma = 10)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[95]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = linear, C = 0.1, gamma = 100
from sklearn.svm import SVC
svm_clf = SVC(kernel='linear', C=0.1, gamma = 100)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[96]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = linear, C = 1.0, gamma = 10
from sklearn.svm import SVC
svm_clf = SVC(kernel='linear', C=1.0, gamma = 10)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[97]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = linear, C = 1.0, gamma = 100
from sklearn.svm import SVC
svm_clf = SVC(kernel='linear', C=1.0, gamma = 100)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[98]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = linear, C = 10.0, gamma = 10
from sklearn.svm import SVC
svm_clf = SVC(kernel='linear', C=10.0, gamma = 10)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[99]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = linear, C = 10.0, gamma = 100
from sklearn.svm import SVC
svm_clf = SVC(kernel='linear', C=10.0, gamma = 100)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[100]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = rbf, C = 0.1, gamma = 10
from sklearn.svm import SVC
svm_clf = SVC(kernel='rbf', C=0.1, gamma = 10)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[101]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = rbf, C = 0.1, gamma = 100
from sklearn.svm import SVC
svm_clf = SVC(kernel='rbf', C=0.1, gamma = 100)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[102]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = rbf, C = 1.0, gamma = 10
from sklearn.svm import SVC
svm_clf = SVC(kernel='rbf', C=1.0, gamma = 10)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[103]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = rbf, C = 1.0, gamma = 100
from sklearn.svm import SVC
svm_clf = SVC(kernel='rbf', C=1.0, gamma = 100)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[104]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = rbf, C = 10.0, gamma = 10
from sklearn.svm import SVC
svm_clf = SVC(kernel='rbf', C=10.0, gamma = 10)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[105]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = rbf, C = 10.0, gamma = 100
from sklearn.svm import SVC
svm_clf = SVC(kernel='rbf', C=10.0, gamma = 100)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[106]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = sigmoid, C = 0.1, gamma = 10
from sklearn.svm import SVC
svm_clf = SVC(kernel='sigmoid', C=0.1, gamma = 10)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[107]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = sigmoid, C = 0.1, gamma = 100
from sklearn.svm import SVC
svm_clf = SVC(kernel='sigmoid', C=0.1, gamma = 100)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[108]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = sigmoid, C = 1.0, gamma = 10
from sklearn.svm import SVC
svm_clf = SVC(kernel='sigmoid', C=1.0, gamma = 10)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[109]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = sigmoid, C = 1.0, gamma = 100
from sklearn.svm import SVC
svm_clf = SVC(kernel='sigmoid', C=1.0, gamma = 100)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[110]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = sigmoid, C = 10.0, gamma = 10
from sklearn.svm import SVC
svm_clf = SVC(kernel='sigmoid', C=10.0, gamma = 10)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[111]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = sigmoid, C = 10.0, gamma = 100
from sklearn.svm import SVC
svm_clf = SVC(kernel='sigmoid', C=10.0, gamma = 100)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[ ]:


# Create a new SVM model
# Use K-fold Cross Validation (k-10)
# Use parameter kernel = poly, C = 0.1, gamma = 10
from sklearn.svm import SVC
svm_clf = SVC(kernel='poly', C=0.1, gamma = 10)

# Train model with cv of 10
cv_scores = cross_val_score(svm_clf, X, y, cv=10)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))

# Putting parameter (kernel = 'poly') takes a very long time 
# So I don't see the result


# In[112]:


# Model with the best score average = 0.
# Parameter // kernel = rbf, C = 0.1, gamma = 10
# Make 2D bar chart to display the accuracy score
svm_clf = SVC(kernel='rbf', C=0.1, gamma = 10)
cv_scores = cross_val_score(svm_clf, X, y, cv=10)

objects = ('1','2','3','4','5','6','7','8','9','10')
y_pos = np.arange(len(objects))

plt.bar(y_pos, cv_scores, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Score')
plt.title('Accuracy Score')

plt.show()


# In[113]:


# Make Confusion Matrix
# Find the confusion matrix of the model with the best score
# Parameter // kernel = rbf, C = 0.1, gamma = 10

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

svm_clf = SVC(kernel='rbf', C=0.1, gamma = 10)

# Train the model
svm_clf.fit(X_train, y_train)

y_pred = svm_clf.predict(X_test)


# In[114]:


data = pd.DataFrame()
data['y_Actual'] =  y_test
data['y_Predicted'] = y_pred


# In[115]:


confusion_matrix = pd.crosstab(data['y_Actual'],data['y_Predicted'],rownames=['Actual'],colnames=['Predicted'],margins=True)

sn.heatmap(confusion_matrix, annot=True)

