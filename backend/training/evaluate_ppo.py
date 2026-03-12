import os
import sys
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(BASE_DIR)

from stable_baselines3 import PPO
from backend.rl.patient_env import PatientEnv


MODEL_PATH = os.path.join(BASE_DIR, "models", "ppo", "ppo_patient_final.zip")


def evaluate(n_episodes=10):
    env = PatientEnv()
    model = PPO.load(MODEL_PATH)

    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {ep+1} Reward:", total_reward)

    print("\n===== RESULT =====")
    print("Mean Reward:", np.mean(rewards))
    print("Std Reward:", np.std(rewards))


if __name__ == "__main__":
    evaluate()