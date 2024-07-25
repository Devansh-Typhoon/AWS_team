

from flask import Flask, request, render_template, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from sklearn.linear_model import LinearRegression
import os
from sklearn.metrics import mean_squared_error, r2_score
import openpyxl
import numpy as np
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx'}


# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# this is for the main page, currently waiting on the html code
@app.route('/', methods = ['GET','POST'])
def index():
    return render_template('index.html')

#this is for the linear regression algo, checking the files and giving back an error
@app.route('/linear_regression', methods = ['GET','POST'])
def linear_regression():
    if 'file' not in request.files:
        return jsonify('error','No file entered')
    file = request.files['file']

    if file.filename == '':
        return jsonify('error', 'No file selected for uploading')
    try:
        df = pd.read_excel(file, usecols="B,E")

        df.head()
        df.info()
        df.describe()

        # CREATING FEATURE MATRIX AND RESPONSE VECTOR
        Date = df.iloc[:, :-1].values  # feature matrix
        Date = pd.to_datetime(df[file, usecols = B])
        Waiting_Time = df.iloc[:, 1].values  # response vector
        Waiting_Time = pd.to_numeric(df['Total Waiting List'], errors='coerce')  # Convert Waiting Time to numeric
        # SPLITTING THE DATA

        Date_train, Date_test, Waiting_Time_train, Waiting_Time_test = train_test_split(Date.values.reshape(-1, 1),
                                                                                        Waiting_Time, test_size=0.30,
                                                                                        random_state=1)

        # FITTING LINEAR REGRESSION MODEL / TRAINING
        regressor = LinearRegression()
        regressor.fit(Date_train, Waiting_Time_train)
        # GETTING THE COEFFICIENTS AND INTERCEPT
        print('Coefficients: ', regressor.coef_)
        print('Intercept: ', regressor.intercept_)

        # PREDICTION OF TEST RESULT
        y_pred = regressor.predict(Date_test)
        print('Predictions:\n', y_pred)

        # COMPARING TEST DATA AND PREDICTED DATA
        comparison_df = pd.DataFrame({"Actual": Waiting_Time_test, "Predicted": y_pred})
        print('Actual test data vs predicted: \n', comparison_df)

        # EVALUATING MODEL METRICS

        print("MSE", mean_squared_error(Waiting_Time_test, y_pred))
        print("RMSE", np.sqrt(mean_squared_error(Waiting_Time_test, y_pred)))
        r2 = r2_score(Waiting_Time_test, y_pred)
        print('Model Score: ', r2)

        # FITTING LINEAR REGRESSION LINE
        '''
        sns.regplot(x='Date', y='Waiting Time', data=df, ci=None, 
                    scatter_kws={'s':100, 'facecolor':'red'})
        '''

        import matplotlib.pyplot as plt
        plt.scatter(Waiting_Time_test, y_pred, alpha=0.5)
        plt.plot([Waiting_Time.min(), Waiting_Time.max()], [Waiting_Time.min(), Waiting_Time.max()], color='red')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Regression')
        plt.show()
        if df.empty:
            return jsonify('error', 'The uploaded file is empty')
#
#     weeks = df[['weeks']].values.reshape(-1,1)
#     total_waiting_list = df[['total_waiting_list']].values
#
#     model = LinearRegression().fit(weeks,total_waiting_list)
#     weeks_new =



    except Exception as E:
        return jsonify({'error': str(E)}), 500







#this just runs the code on the server
if __name__ == '__main__':
    # if not os.path.exists(app.config['UPLOAD_FOLDER']):
    #     os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)