
from flask import Flask, request, render_template, jsonify
import pandas as pd
import matplotlib as mpl
import sklearn as skl
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx'}


# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods = ['GET','POST'])
def index():
    return render_template('index.html')


@app.route('/linear_regression', methods = ['GET','POST'])
def linear_regression():
    if 'file' not in request.files:
        return jsonify('error','No file entered')
    file = request.files['file']

    if file.filename == '':
        return jsonify('error', 'No file selected for uploading')
    try:
        df = pd.read_excel(file)



    if df.empty:
        return jsonify('error', 'The uploaded file is empty')

    if 'weeks' not in df.columns or 'total_waiting_list' not in df.columns
        return jsonify('error', 'Wrong column headings given')







if __name__ == '__main__':
    # if not os.path.exists(app.config['UPLOAD_FOLDER']):
    #     os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)