from flask import Flask, request, render_template,send_file
import pandas as pd
import os
import glob
from classifier import Classifier
from regressor import Regressor
from cleaner import Cleaner

basedir = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(basedir,"uploads")

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
        if request.files['file'].filename == '':
            return render_template('upload.html')
        clean = request.form['cleaned']
        data = pd.read_csv(file_path)
        os.remove(file_path)
        if clean == 'Yes':
            cleaned_data = Cleaner(data)
            cleaned_data.to_csv('cleaned/processed.csv', index=False)

        choice = request.form['choice']
        if choice == 'Regression':
            return render_template('regressor.html')

        if choice =='Classification':
            return render_template('classifier.html')

    return render_template('upload.html')

@app.route('/classifier', methods=['GET', 'POST'])
def classifier():
    if request.method == 'POST':
        clf_selected = request.form['clf_choice']
        df = pd.read_csv('cleaned/processed.csv')
        clf = Classifier(df, clf_selected)
        global best_model
        study, best_model = clf.classify()
        best_parameters = study.best_params
        best_value = study.best_trial.value

        return render_template('result.html', best_parameters=best_parameters, best_value=best_value, best_model=best_model)

    return render_template('classifier.html')

@app.route('/regressor', methods=['GET', 'POST'])
def regressor():
    if request.method == 'POST':
        reg_selected = request.form['reg_choice']
        df = pd.read_csv('cleaned/processed.csv')
        reg = Regressor(df, reg_selected)
        global best_model
        study,best_model = reg.regress()
        best_parameters = study.best_params
        best_value = study.best_trial.value

        return render_template('result.html', best_parameters=best_parameters, best_value=best_value)
    return render_template('regressor.html')


@app.route('/downloads', methods=['GET','POST'])
def download():
    return send_file(best_model, attachment_filename='model.pickle')

if __name__ == '__main__':
    app.run(debug=True)