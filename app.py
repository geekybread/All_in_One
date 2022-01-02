from flask import Flask, request, render_template,jsonify
import pandas as pd
import os
from classifier import Classifier, objective
from cleaner import Cleaner

basedir = os.getcwd()
print(basedir)
UPLOAD_FOLDER = os.path.join(basedir,'uploads/')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            global file_path
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)
    
        clean = request.form['cleaned']
        data = pd.read_csv(file_path)
        if clean == 'Yes':
            cleaned_data = Cleaner(data)
            cleaned_data.to_csv('cleaned/processed.csv')

        choice = request.form['choice']
        if choice == 'Regression':
            return render_template('regressor.html')

        if choice =='Classification':
            return render_template('classifier.html')

    return render_template('upload.html')

@app.route('/classifier', methods=['GET', 'POST'])
def classifier():
    return render_template('classifier.html')

@app.route('/regressor', methods=['GET', 'POST'])
def regressor():
    return render_template('regressor.html')

if __name__ == '__main__':
    app.run(debug=True)