#!/usr/bin/env python -W ignore::Warning
import pickle

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR

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


svr_regressor = SVR(kernel="rbf", C=1.0, epsilon=0.1)
svr_regressor.fit(X_train, y_train)

y_pred_svr = np.round(svr_regressor.predict(X_test))

# Calculate the Root Mean Squared Error (RMSE) and other metrics to evaluate the model
rmse = mean_squared_error(y_test, y_pred_svr, squared=False)
mae = mean_absolute_error(y_test, y_pred_svr)

# Print the metrics
print("Support Vector Regressor Prediction")
print("=============================================")
print(f"The Root Mean Squared Error (RMSE) RF => {rmse:.2f}")
print(f"The Mean Absolute Error (MAE)         => {mae:.2f}")
print("=============================================")
print("GT values vs Prediction")
print("-----------------------")
for gt, pred in zip(y_test, y_pred_svr):
    print(f"GT= {gt:.0f} bpm | RR= {pred:.0f} bpm")
print("")
