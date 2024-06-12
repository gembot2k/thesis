#!/usr/bin/env python -W ignore::Warning
import lib.respiratory as rrm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

dataset = "train"
datafile = "../dataset/radar-" + dataset + "/dataset.txt"
df = pd.read_csv(datafile, sep=" ")

distances = [7, 8]
heights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for distance in distances:
    for height in heights:
        rr_values = []
        gt_values = []

        for idx in df.index:
            iqfile = df["IQFILE"][idx]
            gt = df["RR"][idx]

            complex_signal = rrm.pre_process(dataset, iqfile, distance, height)
            envelope = rrm.signal_process(complex_signal)
            rr = rrm.estimate(envelope, distance, height)

            rr_values.append(rr)
            gt_values.append(gt)

        rmse = mean_squared_error(
            np.array(gt_values), np.array(rr_values), squared=False
        )
        mae = mean_absolute_error(np.array(gt_values), np.array(rr_values))

        print(
            f"[distance={distance}, height={height}] ==> RMSE= {rmse:.2f} | MAE= {mae:.2f}"
        )
