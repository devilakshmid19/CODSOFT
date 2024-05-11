"""automatically generated from 
task-3.ipynb
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tabulate import tabulate

# %%
data=pd.read_csv("IRIS.csv")
data

# %%
data.head()

# %%
data.info()

# %%
data.describe()

# %%
data['species'].value_counts()

# %%
data.isnull().sum()

# %%
X=data.drop('species',axis=1)
y=data['species']

# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

# %%
print(X.shape,X_train.shape,X_test.shape)

# %%
X_train.reset_index(drop=True,inplace=True)
X_test.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)

# %%
data['species'].value_counts()

# %%
data.hist(figsize=(15,10))

# %%
sn.pairplot(data,hue='species',markers='o')
plt.suptitle('Pairplot of iris data',y=1.02)
plt.show()

# %%
plt.figure(figsize=(13,7))
for i in X_train.columns:
    sn.histplot(X_train[i],kde=True)
plt.title('Distribution plot of Features')
plt.show()

# %%
x=data.drop(columns='species')
y=data['species']

# %%
x

# %%
y

# %%
from sklearn.linear_model import LinearRegression
model= LinearRegression()

# %%
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()

# %%
model.fit(X_train,y_train)

# %%
kn_model= KNeighborsClassifier(n_neighbors=3,weights='uniform')
kn_model.fit(X_train,y_train)
pred1_kn=kn_model.predict(X_train)

# %%
plt.figure(figsize=(8,6))
sn.heatmap(confusion_matrix(y_train,pred1_kn),annot=True,fmt='d',cmap='Blues')
plt.title('Confusion Matrix for K-Nearest Neighbors')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# %%
accuracy=model.score(X_train,y_train)*100
print("Accuracy",accuracy)

# %%
from sklearn.tree import DecisionTreeClassifier
dec_model=DecisionTreeClassifier()

# %%
dec_model.fit(X_train,y_train)
pred1_dec=dec_model.predict(X_train)

# %%
plt.figure(figsize=(8,6))
sn.heatmap(confusion_matrix(y_train,pred1_dec),annot=True,fmt='d',cmap='Blues')
plt.title('Confusion Matrix for Decision Tree')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# %%
accuracy=model.score(X_train,y_train)*100
print("Accuracy",accuracy)

# %%
from sklearn.ensemble import RandomForestClassifier
raf_model=RandomForestClassifier()


# %%
raf_model.fit(X_train,y_train)
pred1_raf=raf_model.predict(X_train)

# %%
plt.figure(figsize=(8,6))
sn.heatmap(confusion_matrix(y_train,pred1_raf),annot=True,fmt='d',cmap='Blues')
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# %%
accuracy=model.score(X_train,y_train)*100
print("Accuracy",accuracy)

# %%
from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression()
log_model.fit(X_train,y_train)
pred_log=log_model.predict(X_train)

# %%
plt.figure(figsize=(8,6))
sn.heatmap(confusion_matrix(y_train,pred_log),annot=True,fmt='d',cmap='Blues')
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# %%
plt.figure(figsize=(12,8))
for i in X_train.columns:
    sn.histplot(X_train[i],kde=True)
plt.title('Distribution plot of Features')
plt.show()

# %%
model_results=[
    ["K-Nearest Neighbors",accuracy_score(pred1_kn,y_train)],
    ["Decision Tree",accuracy_score(pred1_dec,y_train)],
    ["Random Forest",accuracy_score(pred1_raf,y_train)],
    ["Logistic Regression",accuracy_score(pred_log,y_train)]
]
print("\nModel Comparison:")
print(tabulate(model_results,headers=["Model","Accuracy"],tablefmt="grid"))


