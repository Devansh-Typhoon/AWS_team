import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(0)
days = np.arange(1, 21)
prices = 100 + days * 2 + np.random.normal(0, 2, days.shape[0])

# Create a DataFrame
data = pd.DataFrame({'Day': days, 'Price': prices})

# Plot the data
plt.scatter(data['Day'], data['Price'], color='blue')
plt.xlabel('Day')
plt.ylabel('Price')
plt.title('Stock Prices Over Time')
plt.show()


# from here is the actual regression code but i just included the other code for reference
# Perform linear regression
X = data[['Day']]
y = data['Price']
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

# Plot the regression line
plt.scatter(data['Day'], data['Price'], color='blue')
plt.plot(data['Day'], predictions, color='red')
plt.xlabel('Day')
plt.ylabel('Price')
plt.title('Stock Prices Over Time with Linear Regression Line')
plt.show()