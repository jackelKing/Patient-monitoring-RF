import os
import sys
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(BASE_DIR)

from stable_baselines3 import PPO


MODEL_PATH = os.path.join(BASE_DIR, "models", "ppo", "ppo_patient_final.zip")


class RLAlertPredictor:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        print("Loading model:", MODEL_PATH)
        self.model = PPO.load(MODEL_PATH)

    def predict(self, state_window):
        """
        state_window shape → (12, 6)
        """

        state_window = np.array(state_window, dtype=np.float32)

        # normalize same way as training (safe fallback)
        state_window = np.nan_to_num(state_window)

        action, _ = self.model.predict(state_window)

        return int(action)