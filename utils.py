# utils.py -- plotting helpers for training diagnostics
import os
import matplotlib.pyplot as plt


def plot_profits(profits, title="Training Episode Rewards", filename="profits.png"):
    os.makedirs("plots", exist_ok=True)
    save_path = os.path.join("plots", filename)
    plt.figure(figsize=(10, 5))
    plt.plot(profits)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved → {save_path}")


def plot_val_curve(val_rewards, filename="val_rewards.png"):
    if not val_rewards:
        return
    steps, rewards = zip(*val_rewards)
    os.makedirs("plots", exist_ok=True)
    save_path = os.path.join("plots", filename)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, rewards, marker="o")
    plt.title("Validation Reward During Training")
    plt.xlabel("Training Step")
    plt.ylabel("Val Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved → {save_path}")
