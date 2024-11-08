import pygame
# Initialize the mixer
pygame.mixer.init()

# Load sounds once and reuse them
SOUNDS = {
    "move": pygame.mixer.Sound("sounds/move.wav"),
    "eat": pygame.mixer.Sound("sounds/eat_dot.wav"),
    "player_death": pygame.mixer.Sound("sounds/player_death.wav"),
    "game_over": pygame.mixer.Sound("sounds/game_over.wav"),
    "power_pellet": pygame.mixer.Sound("sounds/power_pellet_eat.wav"),
    "victory": pygame.mixer.Sound("sounds/victory.wav"),
    "start": pygame.mixer.Sound("sounds/start.wav")
}

# Functions to play sounds
def play_sound(name, loop=False):
    """Play a sound by name with optional looping."""
    sound = SOUNDS.get(name)
    if sound:
        sound.play(-1 if loop else 0)  # -1 loops indefinitely if `loop=True`

def stop_sound(name):
    """Stop a sound by name."""
    sound = SOUNDS.get(name)
    if sound:
        sound.stop()

def stop_all_sounds():
    """Stop all currently playing sounds."""
    pygame.mixer.stop()
