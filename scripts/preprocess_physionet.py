import os
import pandas as pd
import numpy as np
from tqdm import tqdm

RAW_PATH = "../data/raw/physionet2012/set-a"
SAVE_PATH = "../data/processed/physionet_timeseries.csv"

VITALS = ["HR", "SysBP", "DiasBP", "RespRate", "Temp", "SpO2"]

def process_file(file_path):
    df = pd.read_csv(file_path)

    df = df[df["Parameter"].isin(VITALS)]

    df = df.pivot_table(
        index="Time",
        columns="Parameter",
        values="Value",
        aggfunc="mean"
    )

    df = df.sort_index()
    df = df.fillna(method="ffill").fillna(method="bfill")

    return df

def main():
    all_data = []

    files = os.listdir(RAW_PATH)

    for file in tqdm(files):
        file_path = os.path.join(RAW_PATH, file)

        try:
            df = process_file(file_path)
            df["patient_id"] = file.replace(".txt", "")
            all_data.append(df.reset_index())

        except Exception as e:
            print("Skipped:", file)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(SAVE_PATH, index=False)

    print("Saved:", SAVE_PATH)

if __name__ == "__main__":
    main()