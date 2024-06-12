#!/usr/bin/env python -W ignore::Warning
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

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


# Initialize and train the XGBoost Regressor
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)

# Perform predictions on the testing data
y_pred = np.round(xgb.predict(X_test))

# Calculate the Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) to evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

# Print the metrics
print("XGBoost Regressor Prediction")
print("=============================================")
print(f"The Root Mean Squared Error (RMSE)    => {rmse:.2f}")
print(f"The Mean Absolute Error (MAE)         => {mae:.2f}")
print("=============================================")
print("GT values vs Prediction")
print("-----------------------")
for gt, pred in zip(y_test, y_pred):
    print(f"GT= {gt:.0f} bpm | RR= {pred:.0f} bpm")
print("")

# Calculate the average and the difference between predictions and ground truth
mean = np.mean([y_pred, y_test], axis=0)
diff = y_pred - y_test  # Difference between prediction and ground truth
md = np.mean(diff)  # Mean of the difference
sd = np.std(diff, axis=0)  # Standard deviation of the difference

# Creating the Bland-Altman plot
plt.figure(figsize=(10, 6))
plt.scatter(mean, diff, alpha=0.5)
plt.axhline(md, color="gray", linestyle="--")
plt.axhline(md + 1.96 * sd, color="gray", linestyle="--")
plt.axhline(md - 1.96 * sd, color="gray", linestyle="--")

plt.text(np.max(mean), md, "Mean", va="center", ha="right")
plt.text(np.max(mean), md + 1.96 * sd, "+1.96 SD", va="center", ha="right")
plt.text(np.max(mean), md - 1.96 * sd, "-1.96 SD", va="center", ha="right")

plt.title("Bland-Altman Plot")
plt.xlabel("Mean of Ground Truth and Prediction")
plt.ylabel("Difference Between Prediction and Ground Truth")
plt.show()
