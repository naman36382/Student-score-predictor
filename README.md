# Student-score-predictor
This project is a simple web application built with Flask that predicts a student's G3 score based on input features such as absences, study time, G1, and G2. The application includes a dark-themed navigation bar and form with a sleek design.
# Student Board Predictor

This project implements a web application using Flask that predicts a student's final exam score (G3) based on input features such as absences, study time, and grades in the first two exams (G1 and G2). The prediction model is implemented using the Logistic Regression algorithm from scikit-learn, and the web interface is built using Flask, HTML, and CSS.


This file (main.py) is the main script for a Flask web application that predicts a student's final exam score based on input features. The script includes the Flask app setup, route definition, model training, and prediction logic.
# Importing Libraries
### main.py
```python
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings 
warnings.filterwarnings("ignore")
```
- Flask is imported to create the web application.
- pandas and numpy are used for data manipulation.
- train_test_split is used to split the dataset for training and testing.
- LogisticRegression is the machine learning model used for prediction.
- accuracy_score is used to evaluate the model.
- warnings is imported to suppress unnecessary warnings during execution.

# Initializing Flask App
The Flask app is created and assigned to the variable app.
```python
app = Flask(__name__)
```
# Home Route and Function
- The route '/' is defined for both GET and POST methods.
- The home function handles requests and renders the template.
- If a POST request is received, it extracts form data and calls the predict function.
```python
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

```
# Prediction Function
- The predict function reads the dataset, splits it, and trains a Logistic Regression model.
- Input data is prepared for prediction, and the final prediction is returned.
```python
def predict(absences, studytime, G1, G2):
    student_score = pd.read_csv('D:\\naman\Coding\PythonCoding\\randomshi\FlaskApp\student_score.csv')
    X = student_score.drop(columns=['G3', 'useless'], axis=1)
    Y = student_score['G3']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    input_data = np.array([int(absences), int(studytime), int(G1), int(G2)]).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

```
# Running the Flask App
- Checks if the script is being run directly.
- If true, the Flask app is run in debug mode.
- This script, along with the HTML template and CSS file, creates a simple web interface for predicting student exam scores based on input parameters.
```python
if __name__ == '__main__':  
    app.run(debug=True)
```