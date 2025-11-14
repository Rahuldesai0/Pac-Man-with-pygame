import pygame
from settings import *

map_layout = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 73, 71, 72, 0, 83, 67, 79, 82, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [48, 48, 48, 48, 48, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [14, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 21, 22, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 13],
    [9, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 2, 1, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 10],
    [9, 25, 6, 4, 4, 5, 25, 6, 4, 4, 4, 5, 25, 2, 1, 25, 6, 4, 4, 4, 5, 25, 6, 4, 4, 5, 25, 10],
    [9, 26, 2, 0, 0, 1, 25, 2, 0, 0, 0, 1, 25, 2, 1, 25, 2, 0, 0, 0, 1, 25, 2, 0, 0, 1, 26, 10],
    [9, 25, 7, 3, 3, 8, 25, 7, 3, 3, 3, 8, 25, 7, 8, 25, 7, 3, 3, 3, 8, 25, 7, 3, 3, 8, 25, 10],
    [9, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 10],
    [9, 25, 6, 4, 4, 5, 25, 6, 5, 25, 6, 4, 4, 4, 4, 4, 4, 5, 25, 6, 5, 25, 6, 4, 4, 5, 25, 10],
    [9, 25, 7, 3, 3, 8, 25, 2, 1, 25, 7, 3, 3, 5, 6, 3, 3, 8, 25, 2, 1, 25, 7, 3, 3, 8, 25, 10],
    [9, 25, 25, 25, 25, 25, 25, 2, 1, 25, 25, 25, 25, 2, 1, 25, 25, 25, 25, 2, 1, 25, 25, 25, 25, 25, 25, 10],
    [15, 12, 12, 12, 12, 5, 25, 2, 7, 4, 4, 5, 0, 2, 1, 0, 6, 4, 4, 8, 1, 25, 6, 12, 12, 12, 12, 16],
    [0, 0, 0, 0, 0, 9, 25, 2, 6, 3, 3, 8, 0, 7, 8, 0, 7, 3, 3, 5, 1, 25, 10, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 9, 25, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 25, 10, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 9, 25, 2, 1, 0, 6, 12, 12, 27, 27, 12, 12, 5, 0, 2, 1, 25, 10, 0, 0, 0, 0, 0],
    [11, 11, 11, 11, 11, 8, 25, 7, 8, 0, 10, 0, 0, 0, 0, 0, 0, 9, 0, 7, 8, 25, 7, 11, 11, 11, 11, 11],
    [0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0],
    [12, 12, 12, 12, 12, 5, 25, 6, 5, 0, 10, 0, 0, 0, 0, 0, 0, 9, 0, 6, 5, 25, 6, 12, 12, 12, 12, 12],
    [0, 0, 0, 0, 0, 9, 25, 2, 1, 0, 7, 11, 11, 11, 11, 11, 11, 8, 0, 2, 1, 25, 10, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 9, 25, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 25, 10, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 9, 25, 2, 1, 0, 6, 4, 4, 4, 4, 4, 4, 5, 0, 2, 1, 25, 10, 0, 0, 0, 0, 0],
    [14, 11, 11, 11, 11, 8, 25, 7, 8, 0, 7, 3, 3, 5, 6, 3, 3, 8, 0, 7, 8, 25, 7, 11, 11, 11, 11, 13],
    [9, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 2, 1, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 10],
    [9, 25, 6, 4, 4, 5, 25, 6, 4, 4, 4, 5, 25, 2, 1, 25, 6, 4, 4, 4, 5, 25, 6, 4, 4, 5, 25, 10],
    [9, 25, 7, 3, 5, 1, 25, 7, 3, 3, 3, 8, 25, 7, 8, 25, 7, 3, 3, 3, 8, 25, 2, 6, 3, 8, 25, 10],
    [9, 26, 25, 25, 2, 1, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 2, 1, 25, 25, 26, 10],
    [19, 3, 5, 25, 2, 1, 25, 6, 5, 25, 6, 4, 4, 4, 4, 4, 4, 5, 25, 6, 5, 25, 2, 1, 25, 6, 3, 20],    
    [18, 4, 8, 25, 7, 8, 25, 2, 1, 25, 7, 3, 3, 5, 6, 3, 3, 8, 25, 2, 1, 25, 7, 8, 25, 7, 4, 17],
    [9, 25, 25, 25, 25, 25, 25, 2, 1, 25, 25, 25, 25, 2, 1, 25, 25, 25, 25, 2, 1, 25, 25, 25, 25, 25, 25, 10],
    [9, 25, 6, 4, 4, 4, 4, 8, 7, 4, 4, 5, 25, 2, 1, 25, 6, 4, 4, 8, 7, 4, 4, 4, 4, 5, 25, 10],
    [9, 25, 7, 3, 3, 3, 3, 3, 3, 3, 3, 8, 25, 7, 8, 25, 7, 3, 3, 3, 3, 3, 3, 3, 3, 8, 25, 10],
    [9, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 10],
    [15, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 16],
    [28, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

class Map:
    def __init__(self):
        self.textures = {
            'dot': pygame.image.load(TEXTURE_PATHS['dot']),
            'space': pygame.image.load(TEXTURE_PATHS['space']),
            'power_pellet': pygame.image.load(TEXTURE_PATHS['power_pellet']),
            'player_rest': pygame.image.load(TEXTURE_PATHS['player_rest']),
            'player_right': pygame.image.load(TEXTURE_PATHS['player_right']),
            'player_left': pygame.image.load(TEXTURE_PATHS['player_left']),
            'player_up': pygame.image.load(TEXTURE_PATHS['player_up']),
            'player_down': pygame.image.load(TEXTURE_PATHS['player_down']),
            'red_ghost': pygame.image.load(TEXTURE_PATHS['red_ghost']),
            'orange_ghost': pygame.image.load(TEXTURE_PATHS['orange_ghost']),
            'cyan_ghost': pygame.image.load(TEXTURE_PATHS['cyan_ghost']),
            'pink_ghost': pygame.image.load(TEXTURE_PATHS['pink_ghost']),
            'gate': pygame.image.load(TEXTURE_PATHS['gate']),
            '!': pygame.image.load(TEXTURE_PATHS['!']),
        }

        for i in range(65, 91):
            self.textures[chr(i)] = pygame.image.load(TEXTURE_PATHS[chr(i)])

        for i in range(1, 25):
            wall_key = f'wall{i}'
            self.textures[wall_key] = pygame.image.load(TEXTURE_PATHS[wall_key])

        for i in range(10):
            self.textures[f'{i}'] = pygame.image.load(TEXTURE_PATHS[f'{i}'])

    def draw(self, screen):
        for row_idx, row in enumerate(map_layout):
            for col_idx, tile in enumerate(row):
                x, y = col_idx * TILE_SIZE, row_idx * TILE_SIZE
                if 1 <= tile <= 24:
                    screen.blit(self.textures[f"wall{tile}"], (x, y))
                elif tile == 0:
                    screen.blit(self.textures["space"], (x, y))
                elif tile == 25:
                    screen.blit(self.textures["dot"], (x, y))
                elif tile == 26: 
                    screen.blit(self.textures["power_pellet"], (x, y))
                elif tile == 27:
                    screen.blit(self.textures["gate"], (x, y))
                elif tile == 28:
                    screen.blit(self.textures["player_left"], (x, y))
                elif tile == 29:
                    screen.blit(self.textures["red_ghost"], (x, y))
                elif tile == 30:
                    screen.blit(self.textures["orange_ghost"], (x, y))
                elif tile == 31:
                    screen.blit(self.textures["pink_ghost"], (x, y))
                elif tile == 32:
                    screen.blit(self.textures["cyan_ghost"], (x, y))
                else:
                    screen.blit(self.textures[chr(tile)], (x, y))
                

    def get_map_tile(x, y):
        return map_layout[x][y]

