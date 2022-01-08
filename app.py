from flask import Flask, request, render_template,send_file
import pandas as pd
import os
from utils import processor
# from classifier import Classifier
# from regressor import Regressor
from cleaner import Cleaner

from rq import Queue
from worker import conn

q = Queue(connection=conn)

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
        c = 'clf'
        model = request.form['clf_choice']
        global best_model
        df = pd.read_csv('cleaned/processed.csv')
        study, best_model =  q.enqueue(processor, df=df, c=c, model=model)
        best_parameters = study.best_params
        best_value = round(study.best_trial.value,4)*100

        return render_template('result.html', best_parameters=best_parameters, best_value=best_value, best_model=best_model)

    return render_template('classifier.html')

@app.route('/regressor', methods=['GET', 'POST'])
def regressor():
    if request.method == 'POST':
        model = request.form['reg_choice']
        c = 'reg'
        df = pd.read_csv('cleaned/processed.csv')
        global best_model
        study, best_model =  q.enqueue(processor, df,c, model)
        best_parameters = study.best_params
        best_value = round(study.best_trial.value,2)*100

        return render_template('result.html', best_parameters=best_parameters, best_value=best_value)
    return render_template('regressor.html')


@app.route('/downloads', methods=['GET','POST'])
def download():
    return send_file(best_model, attachment_filename='model.pickle')

if __name__ == '__main__':
    app.run(debug=True)