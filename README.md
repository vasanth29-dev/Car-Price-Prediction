import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("dataset/car_data.csv")

# Data preprocessing
data = pd.get_dummies(data, drop_first=True)

# Split features and target
X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Decision Tree Model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predictions
lr_pred = lr_model.predict(X_test)
dt_pred = dt_model.predict(X_test)

# Evaluation
print("Linear Regression MAE:", mean_absolute_error(y_test, lr_pred))
print("Linear Regression R2 Score:", r2_score(y_test, lr_pred))

print("Decision Tree MAE:", mean_absolute_error(y_test, dt_pred))
print("Decision Tree R2 Score:", r2_score(y_test, dt_pred))

# Save model
with open("model/car_price_model.pkl", "wb") as file:
    pickle.dump(dt_model, file)

print("Model saved successfully!")
