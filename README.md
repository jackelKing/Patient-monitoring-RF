#  Edge AI Smart Patient Monitoring & Alert System (RL-Based)

A research-grade, production-ready **Edge AI healthcare monitoring system** that uses **Reinforcement Learning (PPO)** to optimize clinical alerts and reduce alarm fatigue in real-time patient monitoring environments.

---

##  Overview

Traditional patient monitoring systems rely on static thresholds, leading to excessive false alarms and delayed critical alerts.
This project introduces an **adaptive, RL-driven alert optimization system** that:

* Learns optimal alert strategies from ICU vital-sign patterns
* Detects early deterioration signals
* Minimizes unnecessary alerts
* Works in real-time with edge deployment readiness

---

##  Key Features

✔ Reinforcement Learning-based alert optimization (PPO)
✔ Real-time vitals monitoring simulation
✔ Clinical decision-level alert system
✔ End-to-end ML pipeline (data → RL → deployment)
✔ Streamlit dashboard for live demonstration
✔ Production-ready modular backend design

---

##  Dataset

We used the **PhysioNet 2012 Challenge Dataset**, which includes ICU patient time-series vital signs such as:

* Heart Rate (HR)
* Blood Pressure (SysBP, DiasBP)
* Respiratory Rate
* Temperature
* SpO₂

The dataset was processed into RL-ready sliding windows for sequential decision learning.

---

##  System Architecture

```
ICU/Wearable Data
        ↓
Preprocessing Pipeline
        ↓
Sliding Window Generator
        ↓
RL Environment (Gymnasium)
        ↓
PPO Agent Training
        ↓
Inference Engine
        ↓
Streamlit Dashboard
```

---

##  Tech Stack

###  AI / ML

* PyTorch
* Stable-Baselines3 (PPO)
* Gymnasium

###  Backend

* Python
* FastAPI (planned integration)

###  Visualization

* Streamlit
* Matplotlib

###  Data

* NumPy
* Pandas

---

## Workflow

### 1️) Data Processing

* Cleaned ICU vitals
* Filled missing signals
* Generated time-series windows

### 2️) RL Modeling

* Custom Gym environment
* Reward shaping for clinical realism
* PPO training pipeline

### 3️) Evaluation

* Multi-episode evaluation
* Reward stability analysis

### 4️) Deployment

* Real-time inference module
* Interactive Streamlit dashboard

---

##  Dashboard Features

✔ Real-time vitals visualization
✔ RL-based clinical alerts
✔ Simulation mode
✔ Manual vitals entry mode

---

##  How to Run

### 🔹 1. Clone Repo

```bash
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd edge-ai-patient-monitoring
```

### 🔹 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 3. Run Dashboard

```bash
streamlit run streamlit_app/app.py
```

---

##  Results

The PPO agent demonstrated:

✔ strong learning convergence
✔ stable alert decision behavior
✔ consistent reward performance across episodes

---

##  Future Improvements

* Real-time wearable integration
* Multi-patient monitoring
* Edge device deployment (Jetson/Raspberry Pi)
* Transformer-based risk prediction

---

##  Author

**Prem Raga**
* B.Tech CSE — RV Institute of Technology & Management
* B.Sc — IIT Madras (Expected 2027)

---

