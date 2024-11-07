import pygame
import math
from settings import *
from map import *

class Ghost(pygame.sprite.Sprite):
    def __init__(self, x, y, texture, initial_target):
        super().__init__()
        self.image = texture
        self.rect = self.image.get_rect(topleft=(x * TILE_SIZE, y * TILE_SIZE))
        self.target = initial_target
        self.direction = "up"  # Initial movement direction
        self.speed = 1
        self.eyes_mode = False

    def get_position(self):
        return (self.rect.x // TILE_SIZE, self.rect.y // TILE_SIZE)

    def set_target_tile(self, target):
        """Sets the current target tile for the ghost."""
        self.target = target

    def update(self):
        """Update ghost's movement and direction based on its target."""
        # Ensure grid alignment before making direction changes
        if self.is_aligned_with_grid():
            if self.at_junction():
                directions = self.get_valid_directions(map_layout)
                if directions:
                    self.direction = self.choose_best_direction(directions)
        
        # Move in the chosen direction if aligned with grid
        dx, dy = 0, 0
        if self.direction == "up":
            dy = -self.speed
        elif self.direction == "down":
            dy = self.speed
        elif self.direction == "left":
            dx = -self.speed
        elif self.direction == "right":
            dx = self.speed

        # Handle teleportation on row 17
        current_tile_x, current_tile_y = self.get_tile_position()
        if current_tile_y == 17:  # Check if the ghost is on the 17th row
            if current_tile_x == 0 and self.direction == "left":
                # Teleport from leftmost to rightmost
                self.rect.x = (len(map_layout[0]) - 1) * TILE_SIZE
            elif current_tile_x == len(map_layout[0]) - 1 and self.direction == "right":
                # Teleport from rightmost to leftmost
                self.rect.x = 0
            else:
                # Regular movement
                self.rect.x += dx
                self.rect.y += dy
        else:
            # Regular movement if not on the 17th row
            self.rect.x += dx
            self.rect.y += dy


    def is_aligned_with_grid(self):
        """Check if the ghost is aligned with the grid based on tile size."""
        return self.rect.x % TILE_SIZE == 0 and self.rect.y % TILE_SIZE == 0

    def at_junction(self):
        """Checks if the ghost is at a junction."""
        x, y = self.get_tile_position()
        available_paths = sum(
            1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= y + dy < len(map_layout) and 0 <= x + dx < len(map_layout[0]) and 
            map_layout[y + dy][x + dx] not in range(1, 25) and map_layout[y + dy][x + dx] != 27
        )
        return available_paths >= 2


    def get_tile_position(self):
        """Gets the ghost's current position on the map grid."""
        return self.rect.x // TILE_SIZE, self.rect.y // TILE_SIZE

    def get_valid_directions(self, map_layout):
        """Finds valid directions, avoiding reverse moves and walls."""
        directions = {
            "up": (0, -1), "down": (0, 1),
            "left": (-1, 0), "right": (1, 0)
        }
        reverse_direction = {
            "up": "down", "down": "up",
            "left": "right", "right": "left"
        }
        
        valid_directions = {}
        for dir_name, (dx, dy) in directions.items():
            if dir_name == reverse_direction[self.direction]:
                continue  # Avoid reverse movement
            x, y = self.get_tile_position()
            if map_layout[y + dy][x + dx] not in range(1, 25) and map_layout[y + dy][x + dx] != 27:  # No wall
                valid_directions[dir_name] = (x + dx, y + dy)
                
        return valid_directions

    def choose_best_direction(self, directions):
        """Chooses the best direction to minimize Manhattan distance to target with priority order."""
        min_distance = math.inf
        best_direction = self.direction
        tx, ty = self.target
        priority_order = ["up", "down", "left", "right"]

        for dir_name in priority_order:
            if dir_name in directions:
                x, y = directions[dir_name]
                distance = abs(x - tx) + abs(y - ty)
                if distance < min_distance:
                    min_distance = distance
                    best_direction = dir_name

        return best_direction
