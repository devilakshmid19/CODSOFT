"""automatically generated from 
task-1.ipynb
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%
data=pd.read_csv("Titanic-Dataset.csv")

# %%
data.head()

# %%
data.describe()

# %%
data.info()

# %%
data.shape

# %%
data.isnull().sum()

# %%
data['Survived'].value_counts()

# %%
sn.set()

# %%
data['Sex'].value_counts()

# %%
data['P_class'].value_counts()

# %%
data.head()

# %%
data.drop(columns='Cabin',axis=1)

# %%
X=data.drop(columns=['PassengerId','Name','Ticket','Survived'],axis=1)
y=data['Survived']

# %%
X

# %%
y

# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=2)
print(X.shape,X_train.shape,X_test.shape)

# %%
data['Survived'].value_counts()


