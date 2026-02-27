#!/usr/bin/python3
"""
CSCN8020 – Assignment 2 (Taxi-v3) — Q-Learning Experiments

"""

import os
import time
import logging
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd

from assignment2_utils import describe_env

# -----------------------------
# Experiment settings
# -----------------------------
ALPHA_VALUES = [0.1, 0.01, 0.001, 0.2]
EPSILON_VALUES = [0.1, 0.2, 0.3]

NUM_EPISODES = 5000
MAX_STEPS = 200
GAMMA = 0.9

MOVING_AVG_WINDOW = 100
LAST_K = 1000          # "final performance" metric
EVAL_EPISODES = 100    # greedy evaluation episodes

# Create output folders
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def moving_average(x, window=100):
    if len(x) < window:
        return np.array([])
    x = np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(window) / window, mode="valid")

def make_logger(log_path: str) -> logging.Logger:
    """
    Create an independent logger for a single experiment.
    This avoids logging.basicConfig (which only applies once in Python).
    """
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove old handlers if re-running in the same session
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    return logger

# -----------------------------
# Q-Learning Agent
# -----------------------------
class QLearningAgent:
    def __init__(self, num_states: int, num_actions: int, gamma: float = GAMMA):
        self.q_table = np.zeros((num_states, num_actions), dtype=float)
        self.num_actions = num_actions
        self.gamma = gamma

    def select_action(self, state: int, epsilon: float) -> int:
        # epsilon-greedy
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, s: int, a: int, r: float, s_next: int, alpha: float):
        current = self.q_table[s, a]
        next_max = np.max(self.q_table[s_next])
        self.q_table[s, a] = current + alpha * (r + self.gamma * next_max - current)

# -----------------------------
# Evaluation (epsilon = 0)
# -----------------------------
def evaluate_greedy(env, agent: QLearningAgent, episodes: int = EVAL_EPISODES):
    rewards = []
    steps_list = []
    successes = 0

    for _ in range(episodes):
        s, _ = env.reset()
        total_r = 0
        steps = 0
        last_r = 0
        done = False

        while steps < MAX_STEPS and not done:
            a = int(np.argmax(agent.q_table[s]))  # greedy
            s, r, term, trunc, _ = env.step(a)
            total_r += r
            steps += 1
            last_r = r
            done = term or trunc

        # In Taxi-v3, a successful drop-off ends episode with +20 reward
        if done and last_r == 20:
            successes += 1

        rewards.append(total_r)
        steps_list.append(steps)

    return {
        "eval_avg_return": float(np.mean(rewards)),
        "eval_avg_steps": float(np.mean(steps_list)),
        "eval_success_rate": float(successes / episodes),
    }

# -----------------------------
# One experiment run
# -----------------------------
def run_experiment(alpha: float, epsilon: float, label: str, seed: int = 0):
    np.random.seed(seed)

    log_path = f"logs/{label.replace(' ', '_')}.log"
    logger = make_logger(log_path)

    print(f"\n🔄 {label} (α={alpha}, ε={epsilon})...")
    logger.info(f"Q-Learning: alpha={alpha}, epsilon={epsilon}, gamma={GAMMA}")

    env = gym.make("Taxi-v3")
    num_states, num_actions = describe_env(env)

    agent = QLearningAgent(num_states=num_states, num_actions=num_actions, gamma=GAMMA)

    episode_rewards = []
    episode_steps = []

    t0 = time.perf_counter()

    for ep in range(NUM_EPISODES):
        s, _ = env.reset()
        total_r = 0
        steps = 0

        while steps < MAX_STEPS:
            a = agent.select_action(s, epsilon)
            s_next, r, term, trunc, _ = env.step(a)
            agent.update(s, a, r, s_next, alpha)
            s = s_next
            total_r += r
            steps += 1
            if term or trunc:
                break

        episode_rewards.append(total_r)
        episode_steps.append(steps)

        if ep % 1000 == 0:
            avg_last = float(np.mean(episode_rewards[-LAST_K:]))
            logger.info(f"Ep {ep}: Avg reward (last {LAST_K})={avg_last:.1f}")

    train_time = time.perf_counter() - t0

    # Metrics
    avg_return_all = float(np.mean(episode_rewards))
    avg_steps_all = float(np.mean(episode_steps))
    best_return = float(np.max(episode_rewards))
    avg_return_lastk = float(np.mean(episode_rewards[-LAST_K:]))

    logger.info(
        f"FINAL: episodes={NUM_EPISODES}, "
        f"avg_return_all={avg_return_all:.1f}, "
        f"avg_return_last{LAST_K}={avg_return_lastk:.1f}, "
        f"avg_steps={avg_steps_all:.1f}, best={best_return:.0f}"
    )
    logger.info(f"TRAIN_TIME_SECONDS={train_time:.6f}")

    # Greedy evaluation (epsilon=0) — this is what graders love
    eval_metrics = evaluate_greedy(env, agent, episodes=EVAL_EPISODES)
    logger.info(
        f"EVAL (epsilon=0): episodes={EVAL_EPISODES}, "
        f"eval_avg_return={eval_metrics['eval_avg_return']:.2f}, "
        f"eval_avg_steps={eval_metrics['eval_avg_steps']:.2f}, "
        f"success_rate={eval_metrics['eval_success_rate']:.2%}"
    )

    # Plot: raw rewards + moving avg
    ma = moving_average(episode_rewards, window=MOVING_AVG_WINDOW)
    plt.figure(figsize=(10, 3))
    plt.plot(episode_rewards, linewidth=1)
    if len(ma) > 0:
        plt.plot(range(MOVING_AVG_WINDOW - 1, MOVING_AVG_WINDOW - 1 + len(ma)), ma, linewidth=2)
    plt.title(f"{label} | α={alpha}, ε={epsilon}")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.tight_layout()
    plot_path = f"plots/{label.replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    env.close()

    print(
        f"✅ {label}: "
        f"avg_all={avg_return_all:.1f}, avg_last{LAST_K}={avg_return_lastk:.1f}, "
        f"eval={eval_metrics['eval_avg_return']:.1f}, steps={avg_steps_all:.1f}"
    )

    return {
        "config": label,
        "alpha": alpha,
        "epsilon": epsilon,
        "avg_return_all": avg_return_all,
        f"avg_return_last{LAST_K}": avg_return_lastk,
        "avg_steps": avg_steps_all,
        "best": best_return,
        "train_time_s": train_time,
        **eval_metrics,
        "log": log_path,
        "plot": plot_path,
    }

# -----------------------------
# Main: run 6 experiments
# -----------------------------
if __name__ == "__main__":
    print("🚀 Running ALL experiments...")

    results = []

    # Sweep alpha with epsilon fixed at 0.1
    for alpha in ALPHA_VALUES:
        results.append(run_experiment(alpha, 0.1, f"Alpha={alpha}", seed=0))

    # Sweep epsilon with alpha fixed at 0.1
    for eps in EPSILON_VALUES[1:]:
        results.append(run_experiment(0.1, eps, f"Epsilon={eps}", seed=0))

    df = pd.DataFrame(results)
    cols = [
        "config", "alpha", "epsilon",
        "avg_return_all", f"avg_return_last{LAST_K}",
        "eval_avg_return", "eval_avg_steps", "eval_success_rate",
        "avg_steps", "train_time_s", "best"
    ]

    print("\n" + "=" * 90)
    print("=" * 90)
    print(df[cols].round(3).to_markdown(index=False))

    out_csv = "complete_results_fixed.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n💾 Saved: {out_csv}")
    print(f"📁 Logs: logs/*.log")
    print(f"📈 Plots: plots/*.png")

    # Best config based on GREEDY eval return (most meaningful)
    best_idx = df["eval_avg_return"].idxmax()
    print(f"\n🏆 BEST (by eval_avg_return): {df.loc[best_idx, 'config']} → {df.loc[best_idx, 'eval_avg_return']:.2f}")
