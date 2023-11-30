from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings 
warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        absences = request.form['absences'] 
        studytime = request.form['studytime']
        G1 = request.form['G1']
        G2 = request.form['G2']
        prediction = predict(absences, studytime, G1, G2)

    return render_template('index.html', prediction=prediction)

def predict(absences, studytime, G1, G2):
    student_score = pd.read_csv('student_score.csv')
    X = student_score.drop(columns=['G3', 'useless'], axis=1)
    Y = student_score['G3']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    input_data = np.array([int(absences), int(studytime), int(G1), int(G2)]).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

if __name__ == '__main__':  
    app.run(debug=True) 
