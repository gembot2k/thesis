#!/usr/bin/env python -W ignore::Warning
import pickle

import lib.respiratory as rrm
import pandas as pd

datafile = "../dataset/radar-train/dataset.txt"
df = pd.read_csv(datafile, sep=" ")

distance = 8
height = 0.8

features = []
labels = []

for idx in df.index:
    iqfile = df["IQFILE"][idx]
    gt = df["RR"][idx]

    complex_signal = rrm.pre_process("train", iqfile, distance, height)
    envelope = rrm.signal_process(complex_signal)
    feature = rrm.feature_extraction(envelope, distance, height)

    features.append(feature)
    labels.append(gt)

data = {"X_train": features, "y_train": labels}

with open("train_data.pkl", "wb") as file:
    pickle.dump(data, file)

print("Training dataset is saved ==> train_data.pkl")
