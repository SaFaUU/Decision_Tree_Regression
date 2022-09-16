#Data Preprocessing
#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv("Position_Salaries.csv")

#Independent Variable Matrix/ Vector
X = dataset.iloc[:,1:2].values

#Making Dependent Variable Matrix/ Vector
y= dataset.iloc[:, 2].values

#Fitting the model to the dataset
#create regressor
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#Predicting Single Value/ new result with regression
y_pred = regressor.predict(np.array(6.5).reshape(1,-1))


#Visualising the Regression Results with higher resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='purple')
plt.title("Salary vs Levels (Decision Tree Regression with 0.01)")
plt.xlabel("Levels")
plt.ylabel("Salary")
plt.show()