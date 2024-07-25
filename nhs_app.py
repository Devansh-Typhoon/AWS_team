'''
import pandas as pd
NHS_DATA = pd.read_csv('NHS_DATA.csv')
print(NHS_DATA.shape())
print(NHS_DATA.info())

X = NHS_DATA.iloc[:,:-1].values # feature matrix
y = NHS_DATA.iloc[:,1].values # response vector

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)




# importing the linearRegression class
from sklearn.linear_model import LinearRegression

regressor = LinearRegression() # instantiate the Linear Regression model
regressor.fit(X_train, y_train) # training the model

import seaborn as sns
sns.regplot(x='Hours', y='Scores', data=stud_scores, ci=None, scatter_kws={'s':100, 'facecolor':'red'})

y_pred = regressor.predict(X_test)
print(y_pred)

comparison_df = pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
print(comparison_df)

import matplotlib.pyplot as plt
sns.scatterplot(x=y_test, y = y_pred, ci=None, s=140)
plt.xlabel('y_test data')
plt.ylabel('Predictions')

'''

# import all the modules we will need
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# NOTE: methods from sklearn.metrics module can be imported as one liner like:
# from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

# LOADING DATASET
NHS_DATA = pd.read_csv('NHS_DATA.csv')
NHS_DATA.head()
NHS_DATA.info()
NHS_DATA.describe()

# CREATING FEATURE MATRIX AND RESPONSE VECTOR
Date = NHS_DATA.iloc[:,:-1].values # feature matrix
Waiting_Time = NHS_DATA.iloc[:,1].values # response vector

# SPLITTING THE DATA
Date_train, Date_test, Waiting_Time_train, Waiting_Time_test = train_test_split(Date, Waiting_Time, test_size=0.30, random_state=1)

# FITTING LINEAR REGRESSION MODEL / TRAINING
regressor = LinearRegression()
regressor.fit(Date_train, Waiting_Time_train)

# GETTING THE COEFFICIENTS AND INTERCEPT
print('Coefficients: ', regressor.coef_)
print('Intercept: ',regressor.intercept_)

# PREDICTION OF TEST RESULT
y_pred = regressor.predict(Date_test)
print('Predictions:\n', y_pred)

# COMPARING TEST DATA AND PREDICTED DATA
comparison_df = pd.DataFrame({"Actual":Waiting_Time_test,"Predicted":y_pred})
print('Actual test data vs predicted: \n', comparison_df)

# EVALUATING MODEL METRICS
print('MAE:', mean_absolute_error(Waiting_Time_test,y_pred))
print("MSE",mean_squared_error(Waiting_Time_test,y_pred))
print("RMSE",np.sqrt(mean_squared_error(Waiting_Time_test,y_pred)))
r2 = r2_score(Waiting_Time_test,y_pred)
print('Model Score: ', r2)

# FITTING LINEAR REGRESSION LINE
'''
sns.regplot(x='Date', y='Waiting Time', data=NHS_DATA, ci=None, 
            scatter_kws={'s':100, 'facecolor':'red'})
'''

import matplotlib.pyplot as plt
plt.scatter(Waiting_Time_test, y_pred, alpha=0.5)
plt.plot([Waiting_Time.min(), Waiting_Time.max()], [Waiting_Time.min(), Waiting_Time.max()], color='red')
