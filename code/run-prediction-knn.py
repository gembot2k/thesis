#!/usr/bin/env python -W ignore::Warning
import pickle

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

# Loading training data
with open("train_data.pkl", "rb") as file:
    train_data = pickle.load(file)
X_train = train_data["X_train"]
y_train = train_data["y_train"]

# Loading testing data
with open("test_data.pkl", "rb") as file:
    test_data = pickle.load(file)
X_test = test_data["X_test"]
y_test = test_data["y_test"]

# Initialize and train the k-Nearest Neighbors Regressor with additional parameters
knn = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30, p=2)
knn.fit(X_train, y_train)

# Perform predictions on the testing data
y_pred = np.round(knn.predict(X_test))

# Calculate the Root Mean Squared Error (RMSE) and other metrics to evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

# Print the metrics
print("k-Nearest Neighbors Regressor Prediction")
print("=============================================")
print(f"The Root Mean Squared Error (RMSE) kNN => {rmse:.2f}")
print(f"The Mean Absolute Error (MAE)          => {mae:.2f}")
print("=============================================")
print("GT values vs Prediction")
print("-----------------------")
for gt, pred in zip(y_test, y_pred):
    print(f"GT= {gt:.0f} bpm | RR= {pred:.0f} bpm")
print("")
