# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Gather information and presence of null in the dataset.

4.From sklearn.tree import DecisionTreeRegressor and fir the model.

5.Find the mean square error and r squared score value of the model.

6.Check the trained model.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: 212221230111
RegisterNumber: S.LOKESH SAI DILEEP 
*/
#import packages
import pandas as pd
df=pd.read_csv("Salary.csv")
df.head()

#checking the data information and null presence
df.info()
df.isnull().sum()

#encoding categorical features to numeric
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"] = le.fit_transform(df["Position"])
df.head()

#assigning x and y 
x = df[["Position","Level"]]
y = df["Salary"]

#splitting data into training and test
#implementing decision tree regressor in training model
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

#calculating mean square error
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

#calculating r square value
r2 = metrics.r2_score(y_test,y_pred)
r2

#testing the model
dt.predict([[5,6]])
```

## Output:
## Initial Dataset:
![image](https://user-images.githubusercontent.com/94883079/201526060-74b1c219-2655-4200-aa4f-f6bd5fc11958.png)
## Dataset information:
![image](https://user-images.githubusercontent.com/94883079/201526102-fa195744-b18c-4697-a447-8a8afc67be31.png)
## Encoded Dataset:
![image](https://user-images.githubusercontent.com/94883079/201526119-dea9a389-8ac2-40ce-addd-99370d8732b9.png)
## Mean Square Error value:
![image](https://user-images.githubusercontent.com/94883079/201526146-5f5321d9-77e3-4c86-a22b-1ff31a58ea6c.png)
## R squared score:
![image](https://user-images.githubusercontent.com/94883079/201526168-4ff4d900-536d-481c-aa23-fba1fc24661d.png)
## Result value of Model when tested:
![image](https://user-images.githubusercontent.com/94883079/201526193-112e64a7-747e-4360-beba-f12458469638.png)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
