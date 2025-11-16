# game.py
import pygame
from settings import *
from map import *
import time
from player import Player
from ghost import Ghost
from dqn_agent import DQNAgent
import numpy as np
import copy
from collections import deque

ORIGINAL_MAP_LAYOUT = copy.deepcopy(map_layout)

class Game:
    def __init__(self, agent=None):
        global map_layout
        map_layout[:] = copy.deepcopy(ORIGINAL_MAP_LAYOUT)

        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pac-Man RL")
        self.clock = pygame.time.Clock()
        self.agent = agent if agent is not None else DQNAgent()

        self.distance_cache = {}
        self.textures = {
            "player_neutral": pygame.image.load(TEXTURE_PATHS["player_rest"]),
            "player_up": pygame.image.load(TEXTURE_PATHS["player_up"]),
            "player_down": pygame.image.load(TEXTURE_PATHS["player_down"]),
            "player_left": pygame.image.load(TEXTURE_PATHS["player_left"]),
            "player_right": pygame.image.load(TEXTURE_PATHS["player_right"]),
            'red_ghost': pygame.image.load(TEXTURE_PATHS['red_ghost']),
            'orange_ghost': pygame.image.load(TEXTURE_PATHS['orange_ghost']),
            'cyan_ghost': pygame.image.load(TEXTURE_PATHS['cyan_ghost']),
            'pink_ghost': pygame.image.load(TEXTURE_PATHS['pink_ghost']),
            'scared_ghost': pygame.image.load(TEXTURE_PATHS['scared_ghost']),
            'ghost_eyes': pygame.image.load(TEXTURE_PATHS['ghost_eyes']),
        }

        self.start_time = time.time()
        self.reset_game()

    def reset_game(self):
        self.map = Map()
        self.player = Player(13, 26, self.textures)
        self.player.score = 0
        self.player.prev_dot_count = 0  # for reward tracking

        self.red_ghost = Ghost(14, 14, self.textures['red_ghost'], (25, 0))
        self.orange_ghost = Ghost(12, 14, self.textures['orange_ghost'], (2, 35))
        self.cyan_ghost = Ghost(15, 14, self.textures['cyan_ghost'], (26, 36))
        self.pink_ghost = Ghost(13, 14, self.textures['pink_ghost'], (2, 0))

        self.all_sprites = pygame.sprite.Group(self.player, self.red_ghost, self.orange_ghost, self.cyan_ghost, self.pink_ghost)

        # Reset flags
        self.game_over = False
        self.victory = False
        self.step_count = 0

    def get_bfs_distances_from_point(self, start_pos, max_distance=None):
        cache_key = (start_pos, max_distance)
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        distances = {}
        queue = deque([(start_pos, 0)])
        distances[start_pos] = 0
        
        while queue:
            (x, y), dist = queue.popleft()
            if max_distance and dist >= max_distance:
                continue
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT:
                    if not 1 <= map_layout[ny][nx] <= 24 and (nx, ny) not in distances:
                        distances[(nx, ny)] = dist + 1
                        queue.append(((nx, ny), dist + 1))
        
        self.distance_cache[cache_key] = distances
        return distances

    def get_state(self):
        self.distance_cache.clear()
        px, py = self.player.get_position()
        ghosts = [self.red_ghost, self.orange_ghost, self.cyan_ghost, self.pink_ghost]
        
        player_distances = self.get_bfs_distances_from_point((px, py), max_distance=30)
        
        ghost_distances = []
        ghost_scared = []
        min_ghost_distance = float('inf')
        
        for ghost in ghosts:
            gx, gy = ghost.get_position()
            dist = player_distances.get((gx, gy), 9999)
            ghost_distances.append(dist)
            min_ghost_distance = min(min_ghost_distance, dist)
            ghost_scared.append(1.0 if (self.player.power_pellet_active and not ghost.eyes_mode) else 0.0)
        
        max_possible_dist = MAP_WIDTH * MAP_HEIGHT
        ghost_distances_normalized = [d / max_possible_dist for d in ghost_distances]
        
        min_dot_distance = float('inf')
        for y in range(len(map_layout)):
            for x in range(len(map_layout[0])):
                if map_layout[y][x] == 25:
                    dot_dist = player_distances.get((x, y), float('inf'))
                    min_dot_distance = min(min_dot_distance, dot_dist)
        min_dot_distance_normalized = min_dot_distance / max_possible_dist if min_dot_distance != float('inf') else 1.0
        
        danger_up = danger_down = danger_left = danger_right = 0.0
        for i, ghost in enumerate(ghosts):
            if self.player.power_pellet_active and not ghost.eyes_mode:
                continue
            current_dist = ghost_distances[i]
            if current_dist > 15: continue
            gx, gy = ghost.get_position()
            if py > 0 and map_layout[py-1][px] != 1:
                if abs(gx - px) + abs(gy - (py-1)) < abs(gx - px) + abs(gy - py):
                    danger_up += 1.0
            if py < MAP_HEIGHT-1 and map_layout[py+1][px] != 1:
                if abs(gx - px) + abs(gy - (py+1)) < abs(gx - px) + abs(gy - py):
                    danger_down += 1.0
            if px > 0 and map_layout[py][px-1] != 1:
                if abs(gx - (px-1)) + abs(gy - py) < abs(gx - px) + abs(gy - py):
                    danger_left += 1.0
            if px < MAP_WIDTH-1 and map_layout[py][px+1] != 1:
                if abs(gx - (px+1)) + abs(gy - py) < abs(gx - px) + abs(gy - py):
                    danger_right += 1.0
        
        max_danger = max(danger_up, danger_down, danger_left, danger_right, 1.0)
        danger_up /= max_danger
        danger_down /= max_danger
        danger_left /= max_danger
        danger_right /= max_danger
        
        state = np.array([
            px / MAP_WIDTH,
            py / MAP_HEIGHT,
            min_dot_distance_normalized,
            *ghost_distances_normalized,
            *ghost_scared,
            danger_up, danger_down, danger_left, danger_right,
            float(self.player.power_pellet_active),
        ], dtype=np.float32)
        return state

    def get_reward(self):
        px, py = self.player.get_position()
        reward = 0.0

        # === CRITICAL: End episode on ANY death ===
        if self.player.just_died:
            self.game_over = True
            return -150.0  # One-time death penalty

        if self.victory:
            return +1000.0

        # === Progress rewards ===
        dots_eaten = self.player.count_dot - getattr(self.player, 'prev_dot_count', 0)
        reward += dots_eaten * 10.0
        self.player.prev_dot_count = self.player.count_dot

        if self.player.just_ate_power_pellet:
            reward += 50
        if self.player.just_ate_ghost:
            reward += 200

        # === Distance-based shaping ===
        player_distances = self.get_bfs_distances_from_point((px, py))
        min_ghost_dist = min([player_distances.get(g.get_position(), 9999) for g in [self.red_ghost, self.orange_ghost, self.cyan_ghost, self.pink_ghost]])
        min_dot_dist = min([player_distances.get((x,y), 9999) for y in range(MAP_HEIGHT) for x in range(MAP_WIDTH) if map_layout[y][x] == 25], default=0)

        if self.player.power_pellet_active:
            # CHASE GHOSTS!
            reward += 30.0 / (min_ghost_dist + 1)
        else:
            # RUN AWAY!
            if min_ghost_dist < 6:
                reward -= 40.0 / (min_ghost_dist + 1)

        # Always move toward dots
        if min_dot_dist > 0:
            reward += 3.0 / (min_dot_dist + 1)

        # Small living penalty
        reward -= 0.6

        # Wall crash penalty
        nx = px + int(self.player.next_direction.x)
        ny = py + int(self.player.next_direction.y)
        if not (0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT) or 1 <= map_layout[ny][nx] <= 24:
            reward -= 8

        return reward

    def reset(self):
        self.reset_game()
        return self.get_state()

    def step(self, action):
        # Map action to direction
        dirs = ["up", "down", "left", "right"]
        self.player.set_direction(dirs[action])

        # Update game
        self.all_sprites.update()

        # Check victory
        if self.player.count_dot >= 242 and self.player.count_power >= 4:
            self.victory = True

        next_state = self.get_state()
        reward = self.get_reward()
        done = self.game_over or self.victory or self.step_count >= 2000
        self.step_count += 1

        return next_state, reward, done, {}