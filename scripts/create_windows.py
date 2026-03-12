import pandas as pd
import numpy as np
from tqdm import tqdm

INPUT_PATH = "../data/processed/physionet_timeseries.csv"
SAVE_PATH = "../data/processed/physionet_rl_windows.npz"

WINDOW_SIZE = 12
STEP = 1

FEATURES = ["HR", "SysBP", "DiasBP", "RespRate", "Temp", "SpO2"]

def create_windows(df):
    states = []

    # ensure all feature columns exist
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    # fill missing globally
    df[FEATURES] = df[FEATURES].fillna(method="ffill").fillna(method="bfill")

    patient_ids = df["patient_id"].unique()

    for pid in tqdm(patient_ids):
        pdf = df[df["patient_id"] == pid].sort_values("Time")

        values = pdf[FEATURES].values

        for i in range(WINDOW_SIZE, len(values)):
            window = values[i-WINDOW_SIZE:i]
            states.append(window)

    return np.array(states)

def main():
    df = pd.read_csv(INPUT_PATH)

    states = create_windows(df)

    np.savez_compressed(SAVE_PATH, states=states)

    print("Saved:", SAVE_PATH)
    print("Shape:", states.shape)

if __name__ == "__main__":
    main()