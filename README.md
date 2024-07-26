This program is a linear regression algorithm which can be used to input an Excel file and receive a prediction for 15 weeks on what the NHS waiting list is likely to be.
The linear regression model has been trained and has a score of 0.907, and therefore is extremely reliable.

Languages Used:
HTML, Python

Libraries Required:
Matplotlib
Pandas
Scikit-Learn
Numpy
Flask

How to Use:
Press Upload File
Select a .xlsx file with Column Headings 'Week Ending' and 'Total Waiting List'
A graph will then come up on your screen with the blue dots representing existing data, and the red line extending past the final date provided, predicting what the future waiting list size will be
There must be 13 lines above the column headings otherwise, under nhs_app.py, change the line df = pd.read_excel(data_file,skiprows=13), and edit the number of rows above the column headings
To predict for a longer duration of time change the line  new_dates = pd.date_range(start=Date.max() + pd.Timedelta(days=7), periods=15, freq='W') and edit the period to a greater or smaller value

NOTE:
The further in the future you predict for, the more likely this data is to be unreliable, as there are many external variables that could greatly impact this


