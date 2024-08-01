from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load and preprocess data
file_path = r"C:\Users\suhas\Downloads\archive\insurance.csv"
insurance_dataset = pd.read_csv(file_path)

def preprocess_data(data):
    data.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
    data.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
    data.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)
    return data

insurance_dataset = preprocess_data(insurance_dataset)

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)

# Initialize and train models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor()
}

for name, model in models.items():
    model.fit(X_train, Y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model']
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = int(request.form['smoker'])
    region = int(request.form['region'])
    
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    
    # Select model based on user input
    model = models.get(model_name)
    if model:
        prediction = model.predict(input_data)[0]
        return render_template('index.html', prediction_text=f'The predicted insurance cost is USD {prediction:.2f}')
    else:
        return render_template('index.html', prediction_text='Model not found.')

if __name__ == '__main__':
    app.run(debug=True)
