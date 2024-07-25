import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import flask as fl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import openpyxl

data = pd.read_excel("C:/Users/latif/OneDrive/PythonPrograms/NHS.xlsx", usecols="B,E")
complete_data = data.iloc[15:]


print(complete_data)
x = data.iloc[2:3, 15:]
y = data.iloc[5:6, 15:]
plt.scatter(x, y)
plt.title("NHS Records Data")
plt.xlabel("Date")
plt.ylabel("Number Of Waitlist Patients")
plt.show()


#mydataset = {
#  'cars': ["BMW", "Volvo", "Ford"],
#   'passings': [3, 7, 2]
#}

#myvarDataframe = pd.DataFrame(mydataset)
#myvarSeries = pd.Series(mydataset)

#print(myvarDataframe)
#print()
#print(myvarSeries)
