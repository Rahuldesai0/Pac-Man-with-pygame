import pygame
from settings import TILE_SIZE
from map import Map, map_layout

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y, textures):
        super().__init__()
        self.textures = textures
        self.image = self.textures["player_neutral"]
        self.rect = self.image.get_rect(topleft=(x * TILE_SIZE, y * TILE_SIZE))
        
        self.direction = pygame.Vector2(0, 0)
        self.current_texture = "neutral"
        self.frame_count = 0
        self.score = 0

    def update(self):
        current_tile_x = self.rect.x // TILE_SIZE
        current_tile_y = self.rect.y // TILE_SIZE

        if self.direction.x > 0:
            next_tile_x = (self.rect.x + self.rect.width) // TILE_SIZE
        elif self.direction.x < 0:
            next_tile_x = (self.rect.x - 1) // TILE_SIZE
        else:
            next_tile_x = current_tile_x

        if self.direction.y > 0:
            next_tile_y = (self.rect.y + self.rect.height) // TILE_SIZE
        elif self.direction.y < 0:
            next_tile_y = (self.rect.y - 1) // TILE_SIZE
        else:
            next_tile_y = current_tile_y

        if current_tile_y == 17:
            if next_tile_x < 0:
                self.rect.x = (len(map_layout[0]) - 1) * TILE_SIZE
                next_tile_x = len(map_layout[0]) - 1
            elif next_tile_x >= len(map_layout[0]):
                self.rect.x = 0
                next_tile_x = 0

        if not (0 <= next_tile_x < len(map_layout[0]) and 0 <= next_tile_y < len(map_layout)):
            return

        tile_value = map_layout[next_tile_y][next_tile_x]

        if not (1 <= tile_value <= 24 or tile_value == 27) :
            self.rect.x += self.direction.x * TILE_SIZE // 16
            self.rect.y += self.direction.y * TILE_SIZE // 16
            
            if tile_value in [25, 26]:
                score_increment = 10 if tile_value == 25 else 50
                self.score += score_increment
                map_layout[next_tile_y][next_tile_x] = 0
                
            for i in range(6):
                s = str(self.score).zfill(6)
                map_layout[2][i] = ord(s[i])
                

        self.frame_count += 1
        if self.frame_count >= 10:
            self.frame_count = 0
            if self.current_texture == "neutral":
                if self.direction.y < 0:
                    self.image = self.textures["player_up"]
                elif self.direction.y > 0:
                    self.image = self.textures["player_down"]
                elif self.direction.x < 0:
                    self.image = self.textures["player_left"]
                elif self.direction.x > 0:
                    self.image = self.textures["player_right"]
                self.current_texture = "directional"
            else:
                self.image = self.textures["player_neutral"]
                self.current_texture = "neutral"

    def set_direction(self, direction):
        if direction == "up":
            self.direction = pygame.Vector2(0, -1)
        elif direction == "down":
            self.direction = pygame.Vector2(0, 1)
        elif direction == "left":
            self.direction = pygame.Vector2(-1, 0)
        elif direction == "right":
            self.direction = pygame.Vector2(1, 0)
