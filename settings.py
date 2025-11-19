import os

rl_mode = False
MAP_WIDTH = 28
MAP_HEIGHT = 36
TILE_SIZE = 16

SCREEN_WIDTH = MAP_WIDTH * TILE_SIZE
SCREEN_HEIGHT = MAP_HEIGHT * TILE_SIZE

BACKGROUND_COLOR = (0, 0, 0)

TEXTURES_DIR = os.path.join(os.path.dirname(__file__), "textures")
TEXTURE_PATHS = {
    "dot": os.path.join(TEXTURES_DIR, "dot.png"),
    "space": os.path.join(TEXTURES_DIR, "blank.png"),
    "power_pellet": os.path.join(TEXTURES_DIR, "power_pellet.png"),
    "player_rest": os.path.join(TEXTURES_DIR, "pacman_neutral.png"),
    "player_right": os.path.join(TEXTURES_DIR, "pacman_right.png"),
    "player_left": os.path.join(TEXTURES_DIR, "pacman_left.png"),
    "player_up": os.path.join(TEXTURES_DIR, "pacman_up.png"),
    "player_down": os.path.join(TEXTURES_DIR, "pacman_down.png"),
    "red_ghost": os.path.join(TEXTURES_DIR, "red_ghost.png"),
    "orange_ghost": os.path.join(TEXTURES_DIR, "orange_ghost.png"),
    "cyan_ghost": os.path.join(TEXTURES_DIR, "cyan_ghost.png"),
    "pink_ghost": os.path.join(TEXTURES_DIR, "pink_ghost.png"),
    "scared_ghost": os.path.join(TEXTURES_DIR, "scared_ghost.png"),
    "ghost_eyes": os.path.join(TEXTURES_DIR, "ghost_eyes.png"),
    "gate": os.path.join(TEXTURES_DIR, "gate.png"),
    "!": os.path.join(TEXTURES_DIR, "exclamation.png"),
}

for i in range(65, 91):
    TEXTURE_PATHS[chr(i)] = os.path.join(TEXTURES_DIR, f'{chr(i)}.png')

for i in range(1, 25):
    TEXTURE_PATHS[f'wall{i}'] = os.path.join(TEXTURES_DIR, f'wall_{i}.png')

for i in range(10):    
    TEXTURE_PATHS[f'{i}'] = os.path.join(TEXTURES_DIR, f'{i}.png')