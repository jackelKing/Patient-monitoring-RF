import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "physionet_rl_windows.npz")


class PatientEnv(gym.Env):
    def __init__(self):
        super(PatientEnv, self).__init__()

        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

        print("Loading dataset from:", DATA_PATH)

        data = np.load(DATA_PATH)
        states = data["states"]

        # =========================
        # 🔥 CLEAN STATES (CRITICAL)
        # =========================

        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

        # normalize (stable training)
        mean = states.mean()
        std = states.std() + 1e-8
        states = (states - mean) / std

        self.states = states.astype(np.float32)

        self.index = 0

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-5,
            high=5,
            shape=(12, 6),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.index = np.random.randint(0, len(self.states) - 1)
        return self.states[self.index], {}

    def step(self, action):
        state = self.states[self.index]

        risk_score = np.mean(state[-1])

        reward = 0

        if risk_score > 0.5:
            if action == 2:
                reward = 12
            elif action == 1:
                reward = 6
            else:
                reward = -10
        else:
            if action == 0:
                reward = 4
            else:
                reward = -5

        self.index += 1
        done = self.index >= len(self.states) - 1

        next_state = self.states[self.index]

        return next_state, reward, done, False, {}