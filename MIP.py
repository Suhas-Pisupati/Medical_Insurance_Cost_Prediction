import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# loading the data from csv file to a Pandas DataFrame
file_path = r"C:\Users\suhas\Downloads\archive\insurance.csv"
insurance_dataset = pd.read_csv(file_path)

# preprocessing and encoding
insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

# scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=2)

# initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regression': DecisionTreeRegressor(),
    'Random Forest Regression': RandomForestRegressor()
}

# fit models and evaluate
for name, model in models.items():
    model.fit(X_train, Y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    r2_train = metrics.r2_score(Y_train, train_pred)
    r2_test = metrics.r2_score(Y_test, test_pred)
    mae = metrics.mean_absolute_error(Y_test, test_pred)
    mse = metrics.mean_squared_error(Y_test, test_pred)
    rmse = np.sqrt(mse)
    
    print(f"Model: {name}")
    print(f"Training R^2: {r2_train:.4f}")
    print(f"Test R^2: {r2_test:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print("-" * 40)
