#!/usr/bin/env python -W ignore::Warning
import pickle

import pandas as pd

import lib.respiratory as rrm

dataset = "test"
datafile = "./radar-test/dataset.txt"
df = pd.read_csv(datafile, sep=" ")

distance = 8
height = 0.8

features = []
labels = []

for idx in df.index:
    iqfile = df["IQFILE"][idx]
    gt = df["RR"][idx]

    complex_signal = rrm.pre_process("test", iqfile, distance, height)
    envelope = rrm.signal_process(complex_signal)
    feature = rrm.feature_extraction(envelope, distance, height)

    features.append(feature)
    labels.append(gt)

data = {"X_test": features, "y_test": labels}

with open("test_data.pkl", "wb") as file:
    pickle.dump(data, file)

print("Test dataset is saved ==> test_data.pkl")
