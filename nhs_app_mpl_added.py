
from flask import Flask, request, render_template, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from sklearn.linear_model import LinearRegression
import os
from sklearn.metrics import mean_squared_error, r2_score
import openpyxl

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
        complete_data = df.iloc[15:]

        print(complete_data)
        x = data.iloc[15:, 2:3]
        y = data.iloc[15:, 5:6]
        plt.scatter(x, y)
        plt.title("NHS Records Data")
        plt.xlabel("Date")
        plt.ylabel("Number Of Waitlist Patients")
        plt.show()

        if df.empty:
            return jsonify('error', 'The uploaded file is empty')
#chosen axes names are weeks and total waiting list
        if 'weeks' not in df.columns or 'total_waiting_list' not in df.columns:
            return jsonify('error', 'Wrong column headings given')
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
