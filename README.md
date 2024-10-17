![{C78DEC8D-83EF-41DD-AEF8-23DE13E69687}](https://github.com/user-attachments/assets/618b13e6-c900-4c92-a7ea-3d72ece19aec)# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('student_scores.csv')
print(df)

# Check the first and last rows of the dataset
print(df.head(0))  # Display column names (empty dataset for 0 rows)
print(df.tail(0))  # Same for tail

# Display the first and last 5 rows for reference
print(df.head())
print(df.tail())

# Split the data into features (X) and target (y)
X = df.iloc[:, :-1].values  # Assuming the last column is the target (Scores)
y = df.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
regressor = LinearRegression()

# Train the model
regressor.fit(X_train, y_train)

# Predict test results
y_pred = regressor.predict(X_test)

# Display predicted and actual results
print("Predicted values:", y_pred)
print("Actual values:", y_test)

# Plot for training data
plt.scatter(X_train, y_train, color='black')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Plot for test data
plt.scatter(X_test, y_test, color='black')
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Calculate and display errors
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print('Mean Squared Error (MSE) =', mse)
print('Mean Absolute Error (MAE) =', mae)
print('Root Mean Squared Error (RMSE) =',rmse)
```
## Output:
```    Hours  Scores
0     2.5      21
1     5.1      47
2     3.2      27
3     8.5      75
4     3.5      30
5     1.5      20
6     9.2      88
7     5.5      60
8     8.3      81
9     2.7      25
10    7.7      85
11    5.9      62
12    4.5      41
13    3.3      42
14    1.1      17
15    8.9      95
16    2.5      30
17    1.9      24
18    6.1      67
19    7.4      69
20    2.7      30
21    4.8      54
22    3.8      35
23    6.9      76
24    7.8      86
Empty DataFrame
Columns: [Hours, Scores]
Index: []
Empty DataFrame
Columns: [Hours, Scores]
Index: []
   Hours  Scores
0    2.5      21
1    5.1      47
2    3.2      27
3    8.5      75
4    3.5      30
    Hours  Scores
20    2.7      30
21    4.8      54
22    3.8      35
23    6.9      76
24    7.8      86
Predicted values: [83.18814104 27.03208774 27.03208774 69.63323162 59.95115347]
Actual values: [81 30 21 76 62]
```
![{A7763B9C-1269-484C-A0A1-234417B428E3}](https://github.com/user-attachments/assets/799882e0-b9d3-4461-b1b3-c930a7a36e22)
![{44BA293E-5E31-493C-A6D0-757A522ED39D}](https://github.com/user-attachments/assets/56bb5bbf-dcab-4037-9116-5fd1644d163b)
```
Mean Squared Error (MSE) = 18.943211722315272
Mean Absolute Error (MAE) = 3.9207511902099244
Root Mean Squared Error (RMSE) = 4.352380006653288
```



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
