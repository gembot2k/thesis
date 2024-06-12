#!/usr/bin/env python -W ignore::Warning
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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


# Initialize and train the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Perform predictions on the testing data
y_pred_rf = np.round(rf.predict(X_test))

# Calculate the Root Mean Squared Error (RMSE) and other metrics to evaluate the model
rmse = mean_squared_error(y_test, y_pred_rf, squared=False)
mae = mean_absolute_error(y_test, y_pred_rf)

# Print the metrics
print("RandomForest Regressor Prediction")
print("=============================================")
print(f"The Root Mean Squared Error (RMSE) RF => {rmse:.2f}")
print(f"The Mean Absolute Error (MAE)         => {mae:.2f}")
print("")

# Initialize and train the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, random_state=42)
gbr.fit(X_train, y_train)

# Perform predictions on the testing data
y_pred_gb = np.round(gbr.predict(X_test))

# Calculate the Root Mean Squared Error (RMSE) and other metrics to evaluate the model
rmse = mean_squared_error(y_test, y_pred_gb, squared=False)
mae = mean_absolute_error(y_test, y_pred_gb)

# Print the metrics
print("GradientBoosting Regressor Prediction")
print("=============================================")
print(f"The Root Mean Squared Error (RMSE)    => {rmse:.2f}")
print(f"The Mean Absolute Error (MAE)         => {mae:.2f}")
print("")

# Initialize and train the XGBoost Regressor
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)

# Perform predictions on the testing data
y_pred_xgb = np.round(xgb.predict(X_test))

# Calculate the Root Mean Squared Error (RMSE) and other metrics to evaluate the model
rmse = mean_squared_error(y_test, y_pred_xgb, squared=False)
mae = mean_absolute_error(y_test, y_pred_xgb)

# Print the metrics
print("XGBoost Regressor Prediction")
print("=============================================")
print(f"The Root Mean Squared Error (RMSE)    => {rmse:.2f}")
print(f"The Mean Absolute Error (MAE)         => {mae:.2f}")


# Function to calculate mean and diff for Bland-Altman plot
def bland_altman_data(y_pred, y_test):
    mean = np.mean([y_pred, y_test], axis=0)
    diff = y_pred - y_test
    return mean, diff


# Calculating data for Bland-Altman plot
mean_rf, diff_rf = bland_altman_data(y_pred_rf, y_test)
mean_gb, diff_gb = bland_altman_data(y_pred_gb, y_test)
mean_xgb, diff_xgb = bland_altman_data(y_pred_xgb, y_test)

# Set font sizes
title_fontsize = 20
label_fontsize = 18
legend_fontsize = 16
tick_fontsize = 16

# Marker size
marker_size = 100

# Creating the plot
plt.figure(figsize=(12, 8))

# Plotting RandomForest predictions
plt.scatter(mean_rf, diff_rf, alpha=0.6, label="Random Forest", marker="^", s=marker_size)

# Plotting GradientBoosting predictions
plt.scatter(mean_gb, diff_gb, alpha=0.6, label="Gradient Boosting", marker="s", s=marker_size)

# Plotting XGBoost predictions
plt.scatter(mean_xgb, diff_xgb, alpha=0.6, label="XGBoost", marker="o", s=marker_size)

# Adding mean diff line
plt.axhline(np.mean(diff_rf), color="red", linestyle="--", linewidth=1.0)
plt.axhline(np.mean(diff_gb), color="green", linestyle="--", linewidth=1.0)
plt.axhline(np.mean(diff_xgb), color="blue", linestyle="--", linewidth=1.0)

# Adding mean diff ± 1.96*SD lines
plt.axhline(
    np.mean(diff_rf) + 1.96 * np.std(diff_rf),
    color="red",
    linestyle=":",
    linewidth=1.0,
)
plt.axhline(
    np.mean(diff_rf) - 1.96 * np.std(diff_rf),
    color="red",
    linestyle=":",
    linewidth=1.0,
)
plt.axhline(
    np.mean(diff_gb) + 1.96 * np.std(diff_gb),
    color="green",
    linestyle=":",
    linewidth=1.0,
)
plt.axhline(
    np.mean(diff_gb) - 1.96 * np.std(diff_gb),
    color="green",
    linestyle=":",
    linewidth=1.0,
)
plt.axhline(
    np.mean(diff_xgb) + 1.96 * np.std(diff_xgb),
    color="blue",
    linestyle=":",
    linewidth=1.0,
)
plt.axhline(
    np.mean(diff_xgb) - 1.96 * np.std(diff_xgb),
    color="blue",
    linestyle=":",
    linewidth=1.0,
)

#plt.title('Bland-Altman Plot for RR Predictions', fontsize=title_fontsize)
plt.xlabel('Mean of GT and Predictions', fontsize=label_fontsize)
plt.ylabel('Difference Between Predictions and GT', fontsize=label_fontsize)
plt.legend(fontsize=legend_fontsize, loc='lower left')
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.tight_layout()
plt.show()
