# Import all the modules we will need
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# LOADING DATASET
df = pd.read_csv('NHS_LIST.csv')

# Displaying the first few rows, info, and descriptive statistics
print(df.head())
print(df.info())
print(df.describe())
df['Total Waiting List'] = df['Total Waiting List'].str.replace(',', '').astype(float)

# CREATING FEATURE MATRIX AND RESPONSE VECTOR
df['Week Ending'] = pd.to_datetime(df['Week Ending'])  # Convert 'Week Ending' to datetime
#df['Total Waiting List'] = pd.to_numeric(df['Total Waiting List'], errors='coerce')  # Convert 'Total Waiting List' to numeric

# Dropping rows with NaN values to avoid errors in fitting
df = df.dropna(subset=['Total Waiting List'])

# Converting datetime to ordinal to use in linear regression
df['Week Ending Ordinal'] = df['Week Ending'].apply(lambda date: date.toordinal())

# Feature matrix and response vector
X = df[['Week Ending Ordinal']].values  # feature matrix
y = df['Total Waiting List'].values  # response vector

print('This is', df)
print('Value of X' , X)
print(y)
# SPLITTING THE DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, shuffle=True)
# FITTING LINEAR REGRESSION MODEL / TRAINING
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# GETTING THE COEFFICIENTS AND INTERCEPT
print('Coefficients: ', regressor.coef_)
print('Intercept: ', regressor.intercept_)

# PREDICTION OF TEST RESULT
y_pred = regressor.predict(X_test)
print('Predictions:\n', y_pred)

# COMPARING TEST DATA AND PREDICTED DATA
comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print('Actual test data vs predicted: \n', comparison_df)

# EVALUATING MODEL METRICS
print('MAE:', mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)
print('Model Score (R^2): ', r2)

# FITTING LINEAR REGRESSION LINE
sns.regplot(x='Week Ending Ordinal', y='Total Waiting List', data=df, ci=None, scatter_kws={'s':100, 'facecolor':'red'})
plt.title('Regression Line')
plt.show()

# PLOTTING PREDICTED VS ACTUAL VALUES
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Regression')
plt.show()
