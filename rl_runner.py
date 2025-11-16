# rl_runner.py - FINAL VERSION (Graph only at the end)
from game import Game
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
import signal
import sys

# Global list to store rewards
episode_rewards = []
agent = DQNAgent(state_size=16, action_size=4)

def signal_handler(sig, frame):
    """Ctrl+C → show final plot and exit"""
    print(f"\n\nTraining stopped by user after {len(episode_rewards)} episodes.")
    show_final_plot()
    sys.exit(0)

def show_final_plot():
    """Draw and display the final training graph"""
    if not episode_rewards:
        print("No data to plot.")
        return

    plt.figure(figsize=(12, 7))
    episodes = range(1, len(episode_rewards) + 1)
    plt.plot(episodes, episode_rewards, color='skyblue', alpha=0.7, label='Total Reward per Episode')
    
    if len(episode_rewards) >= 100:
        moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        plt.plot(episodes[99:], moving_avg, color='red', linewidth=3, label='100-Episode Moving Average')
    
    plt.axhline(y=600, color='green', linestyle='--', linewidth=2, label='Near-Solved Threshold (+600)')
    plt.title("Pac-Man DQN Training Progress", fontsize=18, pad=20)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Register Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

print("Pac-Man DQN Training Started!")
print("Training silently at maximum speed...")
print("Press Ctrl+C anytime to stop and see the final graph.\n")

episode = 0
try:
    while True:
        episode += 1
        game = Game(agent=agent)
        state = game.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = game.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward

        # === Episode done ===
        episode_rewards.append(total_reward)

        # Progress print (minimal)
        avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else total_reward
        print(f"Episode {episode:4d} | Reward: {total_reward:6.1f} | Avg100: {avg:6.1f} | Steps: {game.step_count}")

        if game.victory:
            print("VICTORY! LEVEL CLEARED!")
        if avg > 650 and len(episode_rewards) >= 100:
            print("\nSOLVED! Average reward > 650 for 100 episodes.")
            break

except KeyboardInterrupt:
    pass  # Ctrl+C handled by signal_handler

finally:
    show_final_plot()
    print(f"\nTraining finished. Total episodes: {len(episode_rewards)}")
    if episode_rewards:
        print(f"Best reward: {max(episode_rewards):.1f}")
        print(f"Final 100-ep avg: {np.mean(episode_rewards[-100:]):.1f}")