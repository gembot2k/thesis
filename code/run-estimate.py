#!/usr/bin/env python -W ignore::Warning
import lib.respiratory as rrm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

dataset = "test"
datafile = "../dataset/radar-" + dataset + "/dataset.txt"
df = pd.read_csv(datafile, sep=" ")

distance = 8
height = 0.8

rr_values = []
gt_values = []

for idx in df.index:
    iqfile = df["IQFILE"][idx]
    gt = df["RR"][idx]

    complex_signal = rrm.pre_process(dataset, iqfile, distance, height)
    envelope = rrm.signal_process(complex_signal)
    rr = rrm.estimate(envelope, distance, height)

    print(f"{iqfile} | GT= {gt:.0f} bpm | RR= {rr:.0f} bpm")
    
    gt_values.append(gt)
    rr_values.append(rr)
    
rmse = mean_squared_error(np.array(gt_values), np.array(rr_values), squared=False)
mae = mean_absolute_error(np.array(gt_values), np.array(rr_values))

print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f}")
    
