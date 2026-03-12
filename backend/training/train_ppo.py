import os
import sys

# =========================
# Fix Import Path (IMPORTANT)
# =========================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(BASE_DIR)


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from backend.rl.patient_env import PatientEnv


# =========================
# Paths
# =========================

MODEL_DIR = os.path.join(BASE_DIR, "models", "ppo")
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# Environment
# =========================

env = make_vec_env(PatientEnv, n_envs=1)


# =========================
# PPO Model
# =========================

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/"
)


# =========================
# Checkpoint Callback
# =========================

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=MODEL_DIR,
    name_prefix="ppo_patient"
)


# =========================
# Train
# =========================

model.learn(
    total_timesteps=50000,
    callback=checkpoint_callback
)

model.save(os.path.join(MODEL_DIR, "ppo_patient_final"))

print("Training Complete ✅")