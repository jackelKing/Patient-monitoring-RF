# =============================
# Fix import path (VERY IMPORTANT)
# =============================
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# =============================
# Imports
# =============================
import streamlit as st
import numpy as np
import pandas as pd
import time

from backend.inference.predictor import RLAlertPredictor


# =============================
# Page Setup
# =============================
st.set_page_config(page_title="Edge AI Patient Monitor", layout="wide")

st.title("🧠 Edge AI Smart Patient Monitoring System")

model = RLAlertPredictor()

# =============================
# Sidebar Controls
# =============================
st.sidebar.header("Mode Selection")

mode = st.sidebar.radio("Choose Mode:", ["Simulation", "Manual Entry"])

# =============================
# Layout
# =============================
col1, col2 = st.columns(2)

chart_placeholder = col1.empty()
alert_placeholder = col2.empty()

# ======================================================
# 🔁 SIMULATION MODE
# ======================================================
if mode == "Simulation":

    if st.sidebar.button("Start Simulation"):

        data = np.random.randn(12, 6)

        for step in range(30):

            new_row = np.random.randn(1, 6)
            data = np.vstack([data[1:], new_row])

            df = pd.DataFrame(
                data,
                columns=["HR", "SysBP", "DiasBP", "RespRate", "Temp", "SpO2"]
            )

            chart_placeholder.line_chart(df)

            action = model.predict(data)

            if action == 0:
                alert_placeholder.success("✅ Normal")
            elif action == 1:
                alert_placeholder.warning("⚠ Alert Nurse")
            else:
                alert_placeholder.error("🚨 Escalate Immediately")

            time.sleep(0.5)

# ======================================================
# ✍ MANUAL ENTRY MODE
# ======================================================
else:

    st.sidebar.subheader("Enter Latest Vitals")

    HR = st.sidebar.number_input("Heart Rate", value=80)
    SysBP = st.sidebar.number_input("SysBP", value=120)
    DiasBP = st.sidebar.number_input("DiasBP", value=80)
    RespRate = st.sidebar.number_input("Resp Rate", value=18)
    Temp = st.sidebar.number_input("Temp", value=37.0)
    SpO2 = st.sidebar.number_input("SpO2", value=98)

    # initialize history
    if "history" not in st.session_state:
        st.session_state.history = np.tile(
            [HR, SysBP, DiasBP, RespRate, Temp, SpO2], (12, 1)
        )

    if st.sidebar.button("Update"):

        new_row = np.array([[HR, SysBP, DiasBP, RespRate, Temp, SpO2]])
        st.session_state.history = np.vstack(
            [st.session_state.history[1:], new_row]
        )

    df = pd.DataFrame(
        st.session_state.history,
        columns=["HR", "SysBP", "DiasBP", "RespRate", "Temp", "SpO2"]
    )

    chart_placeholder.line_chart(df)

    action = model.predict(st.session_state.history)

    if action == 0:
        alert_placeholder.success("✅ Normal")
    elif action == 1:
        alert_placeholder.warning("⚠ Alert Nurse")
    else:
        alert_placeholder.error("🚨 Escalate Immediately")