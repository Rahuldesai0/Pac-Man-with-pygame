from game import Game
from dqn_agent import DQNAgent
import time
import pygame
import matplotlib.pyplot as plt

# Create persistent agent across episodes
agent = DQNAgent()

episode = 0
running = True

try:
    while running:
        episode += 1
        print(f"\n=== Starting Episode {episode} ===")

        # Initialize Pygame and Game
        pygame.init()
        game = Game(agent=agent)

        # Run game until done
        game.run()  # blocking loop

        # Episode finished
        print(f"Episode {episode} finished. Total reward: {sum(agent.episode_rewards[-1:]) if agent.episode_rewards else 0:.2f}\n")

        # Clean up game resources
        del game
        pygame.quit()
        time.sleep(1)

except KeyboardInterrupt:
    print("\nTraining interrupted. Plotting results...")

    # Plot episode number vs total reward
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(agent.episode_rewards) + 1), agent.episode_rewards, marker='o', label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Pac-Man DQN Training Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Ensure the plot stays visible
    plt.show(block=True)

    print("Plot displayed successfully. Exiting cleanly.")