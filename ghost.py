import pygame
import random
from settings import *
from map import *

class Ghost(pygame.sprite.Sprite):
    def __init__(self, x, y, texture, target_tile):
        super().__init__()
        self.image = texture
        self.rect = self.image.get_rect(center=(x * TILE_SIZE, y * TILE_SIZE))
        self.direction = pygame.Vector2(1, 0)
        self.speed = 1
        self.target_tile = target_tile

    def scatter_mode_update(self, map_layout):
        current_tile_x = self.rect.x // TILE_SIZE
        current_tile_y = self.rect.y // TILE_SIZE

        if self.is_at_junction(map_layout, current_tile_x, current_tile_y):
            self.choose_direction(map_layout, current_tile_x, current_tile_y)

        if self.can_move(map_layout, current_tile_x, current_tile_y):
            self.rect.x += self.direction.x * self.speed
            self.rect.y += self.direction.y * self.speed

    def is_at_junction(self, map_layout, tile_x, tile_y):
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1)
        ]
        possible_directions = 0
        for dx, dy in directions:
            if 0 <= tile_x + dx < len(map_layout[0]) and 0 <= tile_y + dy < len(map_layout):
                if map_layout[tile_y + dy][tile_x + dx] == 0 or map_layout[tile_y + dy][tile_x + dx] == 27:
                    possible_directions += 1
        return possible_directions > 2

    def choose_direction(self, map_layout, tile_x, tile_y):
        possible_directions = []
        directions = [
            pygame.Vector2(1, 0), pygame.Vector2(-1, 0), pygame.Vector2(0, 1), pygame.Vector2(0, -1)
        ]

        for direction in directions:
            next_x = tile_x + int(direction.x)
            next_y = tile_y + int(direction.y)
            if 0 <= next_x < len(map_layout[0]) and 0 <= next_y < len(map_layout):
                if map_layout[next_y][next_x] == 0 or map_layout[next_y][next_x] == 27:
                    if direction != -self.direction:
                        possible_directions.append(direction)

        if possible_directions:
            min_distance = float('inf')
            best_direction = self.direction
            for direction in possible_directions:
                next_x = tile_x + int(direction.x)
                next_y = tile_y + int(direction.y)
                distance = ((self.target_tile[0] - next_x) ** 2 + (self.target_tile[1] - next_y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    best_direction = direction
            self.direction = best_direction

    def can_move(self, map_layout, tile_x, tile_y):
        next_x = tile_x + int(self.direction.x)
        next_y = tile_y + int(self.direction.y)
        if 0 <= next_x < len(map_layout[0]) and 0 <= next_y < len(map_layout):
            return map_layout[next_y][next_x] not in range(1, 25)
        return False
