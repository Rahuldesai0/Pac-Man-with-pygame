from game import Game
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
import time

agent = DQNAgent(state_size=16, action_size=4)

# Tracking lists
loss_history = []
survival_history = []
score_history = []

def show_final_plot():
    rewards = agent.episode_rewards
    if not rewards:
        print("No rewards recorded.")
        return

    # ------------------ EXISTING REWARD PLOT ------------------
    plt.figure(figsize=(12, 6))
    episodes = range(1, len(rewards) + 1)

    plt.plot(episodes, rewards, label="Episode Reward", linewidth=1.2)

    if len(rewards) >= 100:
        ma = np.convolve(rewards, np.ones(100)/100, mode="valid")
        plt.plot(episodes[99:], ma, color="red", label="Avg100 (Moving Avg)", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------ LOSS PLOT ------------------
    if loss_history:
        plt.figure(figsize=(12, 6))
        plt.plot(loss_history, label="Loss per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ------------------ SURVIVAL TIME ------------------
    if survival_history:
        plt.figure(figsize=(12, 6))
        plt.plot(survival_history, label="Survival Time per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Seconds Alive")
        plt.title("Episode Survival Duration")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ------------------ SCORE PLOT ------------------
    if score_history:
        plt.figure(figsize=(12, 6))
        plt.plot(score_history, label="Score per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title("Player Score per Episode")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


print("Training started. Press Ctrl+C to stop.\n")

try:
    episode_num = 0

    while True:
        episode_num += 1

        # ================= START TIME TRACK ================
        start_time = time.time()

        steps_before = agent.steps

        game = Game(agent=agent)
        game.run()   # your game loop

        steps_after = agent.steps
        steps_this_episode = steps_after - steps_before

        # ================= SURVIVAL TIME ====================
        survival_history.append(time.time() - start_time)

        # Reward for this episode
        reward = agent.episode_rewards[-1]

        # ================= SCORE TRACK ======================
        score_history.append(game.player.score)

        if game.victory:
            print("Victory")

        # Compute avg100
        if len(agent.episode_rewards) >= 100:
            avg100 = np.mean(agent.episode_rewards[-100:])
        else:
            avg100 = np.mean(agent.episode_rewards)

        print(f"Episode {episode_num:4d} | Reward: {reward:7.1f} | "
              f"Avg100: {avg100:7.1f} | Steps: {steps_this_episode}")

        # ================= STORE LOSS =======================
        # agent.loss is assumed to be updated inside replay()
        if hasattr(agent, "last_loss"):
            loss_history.append(agent.last_loss)
        else:
            # fallback if user didn't add loss tracking in agent
            loss_history.append(0)

except KeyboardInterrupt:
    print("\nTraining stopped by user.\n")

finally:
    total_eps = len(agent.episode_rewards)
    best_reward = max(agent.episode_rewards) if agent.episode_rewards else 0
    final_avg100 = (np.mean(agent.episode_rewards[-100:])
                    if len(agent.episode_rewards) >= 100
                    else np.mean(agent.episode_rewards))

    print(f"Training finished. Total episodes: {total_eps}")
    print(f"Best reward: {best_reward:.1f}")
    print(f"Final 100-episode avg: {final_avg100:.1f}")
    show_final_plot()
