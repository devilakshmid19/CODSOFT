"""automatically generated from 
task-4.ipynb
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# %%
data=pd.read_csv('advertising.csv')


# %%
data.head()

# %%
data.info()

# %%
data.describe()

# %%
data.shape

# %%
data.isnull().sum()

# %%
sns.pairplot(data)
plt.suptitle('Pairplot of sales Data',y=1.02)
plt.show()

# %%
plt.figure(figsize=(8,6))
plt.scatter(x='TV', y='Sales', data=data)
plt.title('Scatter plot: TV vs Sales')
plt.show()

# %%
plt.figure(figsize=(8,6))
plt.scatter(x='Radio', y='Sales', data=data)
plt.title('Scatter plot:Radio vs Sales')
plt.show()

# %%
plt.figure(figsize=(10,6))
sns.histplot(data['Sales'],kde=True)
plt.title('Distribution Sales')
plt.show()

# %%
channels=['TV','Radio','Newspaper']
total_spending=[data['TV'].sum(),data['Radio'].sum(),data['Newspaper'].sum()]
plt.figure(figsize=(8,8))
plt.pie(total_spending,labels=channels,autopct='%1.1f%%',startangle=90,colors=['skyblue','darkgreen','blue'])
plt.title('Distribution of Advertising Spending')
plt.show()

# %%
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(),annot=True,cmap='coolwarm',fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# %%
X=data[['TV','Radio','Newspaper']]
Y=data['Sales']

# %%
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score
from tabulate import tabulate

# %%
models = {
"Linear Regression": LinearRegression(),
"Decision Tree": DecisionTreeRegressor(),
"Random Forest": RandomForestRegressor(),
"Support Vector Regression": SVR()
}
model_results = []
for model_name, model in models.items():
    model.fit(X_train, Y_train)
    Y_pred=model.predict(X_test)
    mse =mean_squared_error(Y_test,Y_pred)
    r2 =r2_score(Y_test,Y_pred)
    model_results.append([model_name, mse, r2])
    print(f"\n{model_name} Evaluation:")
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")
    plt.scatter(Y_test,Y_pred)
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title(f'{model_name}- Actual vs Predicted Sales')
    plt.show()

# %%
print("\nModel Comparison")
print(tabulate(model_results,headers=["Model","Mean Squared Error","R2 Score"],tablefmt="grid"))



