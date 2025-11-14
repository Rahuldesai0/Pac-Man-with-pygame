import pygame
from settings import TILE_SIZE
from map import Map, map_layout
import time
import sound_manager

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y, textures):
        super().__init__()
        self.textures = textures
        self.image = self.textures["player_neutral"]
        self.rect = self.image.get_rect(topleft=(x * TILE_SIZE, y * TILE_SIZE))
        
        self.direction = pygame.Vector2(0, 0)
        self.next_direction = pygame.Vector2(0, 0)
        self.score = 0
        self.count_dot = 0
        self.count_power = 0
        self.speed = TILE_SIZE // 8  # Pac-Man's speed
        self.frame_count = 0
        self.current_texture = "neutral"
        self.power_pellet_active = False
        self.power_pellet_start_time = None

        self.just_ate_dot = False
        self.just_ate_power_pellet = False
        self.just_ate_ghost = False
        self.just_died = False
        self.just_reversed = False


    def update(self):
        # Determine current tile position
        tile_x = self.rect.x // TILE_SIZE
        tile_y = self.rect.y // TILE_SIZE

        # Handle teleportation on row 17
        if tile_y == 17:
            if tile_x == 0 and self.direction == pygame.Vector2(-1, 0):
                self.rect.x = (len(map_layout[0]) - 1) * TILE_SIZE  # Teleport to rightmost tile
            elif tile_x == len(map_layout[0])-1 and self.direction == pygame.Vector2(1, 0):
                self.rect.x = 0  # Teleport to leftmost tile

        # If aligned to grid, try to set the new direction
        if self.is_aligned_with_grid():
            if self.can_move(tile_x, tile_y, self.next_direction):
                self.direction = self.next_direction

            # Stop movement if the next tile in the current direction is a wall
            if not self.can_move(tile_x, tile_y, self.direction):
                self.direction = pygame.Vector2(0, 0)

        # Move Pac-Man in the current direction
        self.rect.x += int(self.direction.x) * self.speed
        self.rect.y += int(self.direction.y) * self.speed

        # Update score and eat dots/pellets
        self.handle_scoring(tile_x, tile_y)

        # Handle animation
        self.animate_texture()

        if self.power_pellet_active and time.time() - self.power_pellet_start_time >= 10:
            self.power_pellet_active = False

    def is_aligned_with_grid(self):
        """Check if Pac-Man is aligned to the grid."""
        return self.rect.x % TILE_SIZE == 0 and self.rect.y % TILE_SIZE == 0

    def can_move(self, tile_x, tile_y, direction):
        """Check if Pac-Man can move in the desired direction."""
        next_tile_x = tile_x + int(direction.x)
        next_tile_y = tile_y + int(direction.y)

        # Ensure the coordinates are within map bounds
        if 0 <= next_tile_x < len(map_layout[0]) and 0 <= next_tile_y < len(map_layout):
            # Return True if the tile is not a wall or ghost-only tile
            return map_layout[next_tile_y][next_tile_x] not in range(1, 25) and map_layout[next_tile_y][next_tile_x] != 27
        return False

    def handle_scoring(self, tile_x, tile_y):
        """Update score based on the current tile and consume dots/pellets."""
        # Reset frame-based reward flags
        self.just_ate_dot = False
        self.just_ate_power_pellet = False
        self.just_ate_ghost = False
        self.just_died = False
        self.just_reversed = False

        tile_value = map_layout[tile_y][tile_x]

        if tile_value == 25:  # Dot
            self.score += 10
            self.count_dot += 1
            self.just_ate_dot = True
            # sound_manager.stop_sound('eat')
            # sound_manager.play_sound('eat')
            map_layout[tile_y][tile_x] = 0

        elif tile_value == 26:  # Power Pellet
            self.score += 50
            self.count_power += 1
            self.just_ate_power_pellet = True
            map_layout[tile_y][tile_x] = 0
            # sound_manager.play_sound("power_pellet")
            self.power_pellet_active = True
            self.power_pellet_start_time = time.time()

        for i in range(6):
            s = str(self.score).zfill(6)
            map_layout[2][i] = ord(s[i])

    def animate_texture(self):
        """Animate Pac-Man's texture based on movement direction."""
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
        """Set the next direction based on input."""
        if direction == "up":
            if self.direction == pygame.Vector2(0, 1):
                self.just_reversed = True
            self.next_direction = pygame.Vector2(0, -1)
        elif direction == "down":
            if self.direction == pygame.Vector2(0, -1):
                self.just_reversed = True
            self.next_direction = pygame.Vector2(0, 1)
        elif direction == "left":
            if self.direction == pygame.Vector2(1, 0):
                self.just_reversed = True
            self.next_direction = pygame.Vector2(-1, 0)
        elif direction == "right":
            if self.direction == pygame.Vector2(-1, 0):
                self.just_reversed = True
            self.next_direction = pygame.Vector2(1, 0)

    def get_position(self):
        return (self.rect.x // TILE_SIZE, self.rect.y // TILE_SIZE)
    
    def set_position(self, x, y):  
        self.rect = self.image.get_rect(topleft=(x * TILE_SIZE, y * TILE_SIZE))

    def get_direction(self):
        if self.direction == pygame.Vector2(0, -1): return "up"
        elif self.direction == pygame.Vector2(0, 1): return "down"
        elif self.direction == pygame.Vector2(-1, 0): return "left"
        elif self.direction == pygame.Vector2(1, 0): return "right"

