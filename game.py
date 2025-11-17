import pygame
from settings import *
from map import *
import time
from player import Player
from ghost import Ghost
from dqn_agent import DQNAgent
import numpy as np
import sound_manager
import copy
from collections import deque

ORIGINAL_MAP_LAYOUT = copy.deepcopy(map_layout)

class Game:
    def __init__(self, agent=None):
        global map_layout
        map_layout[:] = copy.deepcopy(ORIGINAL_MAP_LAYOUT)

        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        # sound_manager.play_sound('start')
        pygame.display.set_caption("Pac Man")
        self.clock = pygame.time.Clock()
        self.agent = agent if agent is not None else DQNAgent()
        self.reward = 0.0              # step reward
        self.total_reward = 0.0        # accumulated reward per episode
        self.step_count = 0

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
        self.lives = 3
        self.game_over = False
        self.victory = False
        
        self.reset_game()

    def reset_player(self):
        # Clear the tile at the current position
        px, py = self.player.get_position()
        map_layout[py][px] = 0  # Set the tile back to an empty state
        
        # Reset the player's position
        self.player.set_position(13, 26)
        self.player.set_direction("neutral")  # Reset to a neutral or initial direction

    def reset_game(self):
        """Reset all game variables to start a new game."""
        self.map = Map()  # Reset the map
        self.player = Player(13, 26, self.textures)  # Reset the player
        self.player.score = 0 
        self.total_reward = 0.0
        self.step_count = 0


        # Reset ghosts with their initial positions
        self.red_ghost = Ghost(14, 14, self.textures['red_ghost'], (25, 0))
        self.orange_ghost = Ghost(12, 14, self.textures['orange_ghost'], (2, 35))
        self.cyan_ghost = Ghost(15, 14, self.textures['cyan_ghost'], (26, 36))
        self.pink_ghost = Ghost(13, 14, self.textures['pink_ghost'], (2, 0))

        self.all_sprites = pygame.sprite.Group(self.player, self.red_ghost, self.orange_ghost, self.cyan_ghost, self.pink_ghost)

    def get_bfs_distances_from_point(self, start_pos, max_distance=None):
        """
        Single BFS that returns distances to ALL reachable points from start_pos.
        Much more efficient than calling BFS multiple times.
        """
        cache_key = (start_pos, max_distance)
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        distances = {}
        queue = deque([(start_pos, 0)])
        distances[start_pos] = 0
        
        while queue:
            (x, y), dist = queue.popleft()
            
            # Early exit if we've reached max distance
            if max_distance and dist >= max_distance:
                continue
            
            # Check all 4 directions
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT:
                    # Check if tile is walkable and not visited
                    if not 1 <= map_layout[ny][nx] <= 24 and (nx, ny) not in distances:
                        distances[(nx, ny)] = dist + 1
                        queue.append(((nx, ny), dist + 1))
        
        self.distance_cache[cache_key] = distances
        return distances

    def get_state(self):
        """Get enhanced state representation with optimized pathfinding."""
        # Clear cache at the start of each state calculation
        self.distance_cache.clear()
        
        px, py = self.player.get_position()
        
        ghosts = [self.red_ghost, self.orange_ghost, self.cyan_ghost, self.pink_ghost]
        
        # Single BFS from player position (gets all distances at once)
        player_distances = self.get_bfs_distances_from_point((px, py), max_distance=30)
        
        # Ghost distances and scared status
        ghost_distances = []
        ghost_scared = []
        min_ghost_distance = float('inf')
        
        for ghost in ghosts:
            gx, gy = ghost.get_position()
            dist = player_distances.get((gx, gy), MAP_WIDTH * MAP_HEIGHT)
            ghost_distances.append(dist)
            min_ghost_distance = min(min_ghost_distance, dist)
            ghost_scared.append(1.0 if (self.player.power_pellet_active and not ghost.eyes_mode) else 0.0)
        
        # Normalize ghost distances
        max_possible_dist = MAP_WIDTH * MAP_HEIGHT
        ghost_distances_normalized = [d / max_possible_dist for d in ghost_distances]
        
        # Find nearest dot using the same BFS result
        min_dot_distance = float('inf')
        for y in range(len(map_layout)):
            for x in range(len(map_layout[0])):
                if map_layout[y][x] == 25:
                    dot_dist = player_distances.get((x, y), float('inf'))
                    min_dot_distance = min(min_dot_distance, dot_dist)
        
        if min_dot_distance == float('inf'):
            min_dot_distance_normalized = 1.0
        else:
            min_dot_distance_normalized = min_dot_distance / max_possible_dist
        
        # Directional danger - only check immediate neighbors
        danger_up = danger_down = danger_left = danger_right = 0.0
        
        # For each direction, check if moving brings us closer to dangerous ghosts
        for i, ghost in enumerate(ghosts):
            if self.player.power_pellet_active and not ghost.eyes_mode:
                continue  # Skip scared ghosts
            
            current_ghost_dist = ghost_distances[i]
            if current_ghost_dist > 15:  # Ignore very distant ghosts
                continue
            
            gx, gy = ghost.get_position()
            
            # Check up
            if py > 0 and map_layout[py-1][px] != 1:
                up_dist = player_distances.get((gx, gy), float('inf'))  # Distance from current pos
                # Estimate if up moves us closer (simple heuristic)
                if abs(gx - px) + abs(gy - (py-1)) < abs(gx - px) + abs(gy - py):
                    danger_up += 1.0
            
            # Check down
            if py < MAP_HEIGHT-1 and map_layout[py+1][px] != 1:
                if abs(gx - px) + abs(gy - (py+1)) < abs(gx - px) + abs(gy - py):
                    danger_down += 1.0
            
            # Check left
            if px > 0 and map_layout[py][px-1] != 1:
                if abs(gx - (px-1)) + abs(gy - py) < abs(gx - px) + abs(gy - py):
                    danger_left += 1.0
            
            # Check right
            if px < MAP_WIDTH-1 and map_layout[py][px+1] != 1:
                if abs(gx - (px+1)) + abs(gy - py) < abs(gx - px) + abs(gy - py):
                    danger_right += 1.0
        
        # Normalize danger values
        max_danger = max(danger_up, danger_down, danger_left, danger_right, 1.0)
        danger_up /= max_danger
        danger_down /= max_danger
        danger_left /= max_danger
        danger_right /= max_danger
        
        state = np.array([
            px / MAP_WIDTH,
            py / MAP_HEIGHT,
            min_dot_distance_normalized,
            ghost_distances_normalized[0],  # Red ghost
            ghost_distances_normalized[1],  # Orange ghost
            ghost_distances_normalized[2],  # Cyan ghost
            ghost_distances_normalized[3],  # Pink ghost
            ghost_scared[0],
            ghost_scared[1],
            ghost_scared[2],
            ghost_scared[3],
            danger_up,
            danger_down,
            danger_left,
            danger_right,
            float(self.player.power_pellet_active),
        ], dtype=np.float32)
        return state

    # def get_reward(self):
    #     # Track current & previous positions
    #     px, py = self.player.get_position()
    #     if not hasattr(self, "prev_positions"):
    #         self.prev_positions = deque(maxlen=5)
    #     self.prev_positions.append((px, py))

    #     # Anti-jiggle detection (oscillating between two tiles)
    #     jiggle_penalty = 0
    #     if len(self.prev_positions) >= 4:
    #         if (self.prev_positions[-1] == self.prev_positions[-3] and 
    #             self.prev_positions[-2] == self.prev_positions[-4]):
    #             jiggle_penalty = -12  # harsh penalty to break oscillation

    #     # BFS distances from player
    #     player_distances = self.get_bfs_distances_from_point((px, py), max_distance=MAP_WIDTH*MAP_HEIGHT)

    #     ghosts = [self.red_ghost, self.orange_ghost, self.cyan_ghost, self.pink_ghost]
    #     total_ghost_distance = 0
    #     min_ghost_distance = float('inf')
    #     scared_ghosts_nearby = 0

    #     for ghost in ghosts:
    #         gx, gy = ghost.get_position()
    #         ghost_dist = player_distances.get((gx, gy), MAP_WIDTH * MAP_HEIGHT)
    #         total_ghost_distance += ghost_dist
    #         min_ghost_distance = min(min_ghost_distance, ghost_dist)
    #         if self.player.power_pellet_active and ghost_dist < 15 and not ghost.eyes_mode:
    #             scared_ghosts_nearby += 1

    #     # Find nearest dot using BFS
    #     min_dot_distance = float('inf')
    #     for (x, y), dist in player_distances.items():
    #         if map_layout[y][x] == 25:
    #             min_dot_distance = min(min_dot_distance, dist)
    #     if min_dot_distance == float('inf'):
    #         min_dot_distance = 0

    #     # Find nearest power pellet using BFS
    #     min_power_distance = float('inf')
    #     for (x, y), dist in player_distances.items():
    #         if map_layout[y][x] == 26:
    #             min_power_distance = min(min_power_distance, dist)
    #     if min_power_distance == float('inf'):
    #         min_power_distance = 0

    #     # Initialize reward accumulator
    #     reward = 0

    #     # --- Event-based rewards ---
    #     if self.player.just_ate_dot:
    #         reward += 12
    #     if self.player.just_ate_power_pellet:
    #         reward += 60   # 50 base + small bonus
    #     if self.player.just_ate_ghost:
    #         reward += 200
    #     if self.player.just_died:
    #         reward -= 350
    #     if self.game_over:
    #         reward -= 350
    #     elif self.victory:
    #         reward += 500

    #     # --- Distance-based rewards ---
    #     if self.player.power_pellet_active:
    #         if scared_ghosts_nearby > 0:
    #             reward += 15.0 / (min_ghost_distance + 1)
    #         else:
    #             reward -= 2.0
    #     else:
    #         if min_ghost_distance < 5:
    #             reward -= 40.0 / (min_ghost_distance + 1)
    #         elif min_ghost_distance < 10:
    #             reward -= 15.0 / (min_ghost_distance + 1)
    #         else:
    #             reward -= 3.0 / (total_ghost_distance + 1)

    #     # Encourage approaching dots/pellets
    #     if min_dot_distance > 0:
    #         reward += 1.5 / (min_dot_distance + 1)
    #     if min_power_distance > 0:
    #         reward += 7.5 / (min_power_distance + 1)

    #     # --- Movement penalties / smoothing ---
    #     # Step cost
    #     reward -= 5
    #     # Penalty for reversing direction too often
    #     if self.player.just_reversed:
    #         reward -= 6
    #     # Penalty for standing still
    #     if len(self.prev_positions) >= 2 and self.prev_positions[-1] == self.prev_positions[-2]:
    #         reward -= 8
    #     # Penalty for jiggling (A↔B↔A↔B)
    #     reward += jiggle_penalty

    #     # Penalty if next_direction faces wall
    #     nx, ny = px + int(self.player.next_direction.x), py + int(self.player.next_direction.y)
    #     if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT:
    #         if 1 <= map_layout[ny][nx] <= 24:
    #             reward -= 10

    #     # --- Small positive reward for actual movement ---
    #     if len(self.prev_positions) >= 2 and self.prev_positions[-1] != self.prev_positions[-2]:
    #         reward += 1.5

    #     reward -= 0.5
    #     return reward

    def get_reward(self):
        """
        Stable Pac-Man reward function optimized for fast learning
        without hurting convergence quality.
        """

        reward = 0

        # --- TERMINAL EVENTS (highest priority) ---
        if self.player.just_died:
            return -250   # strong negative – prevents suicide farming
        if self.game_over:
            return -400   # finishing with loss is worse than dying once
        if self.victory:
            return +1500  # strongest single reward


        # --- POSITIVE EVENTS ---
        if self.player.just_ate_dot:
            reward += 8       # encourages clearing maze fast

        if self.player.just_ate_power_pellet:
            reward += 40      # encourages strategic pellet timing

        if self.player.just_ate_ghost:
            reward += 200     # high reward but not insane (keeps stability)


        # --- SURVIVAL + MOVEMENT REWARDS ---
        # mild survival reward
        reward += 0.5

        # reward for actually changing tiles (not oscillating)
        px, py = self.player.get_position()
        if not hasattr(self, "last_pos"):
            self.last_pos = (px, py)
        if (px, py) != self.last_pos:
            reward += 1.0     # encourages exploration
        else:
            reward -= 2.0     # discourages standing still

        # store for next step
        self.last_pos = (px, py)


        # --- GHOST DISTANCE SHAPING (stabilizing component) ---
        ghosts = [self.red_ghost, self.orange_ghost, self.cyan_ghost, self.pink_ghost]
        player_pos = (px, py)
        dists = self.get_bfs_distances_from_point(player_pos)

        min_ghost_dist = min(
            dists.get(ghost.get_position(), 9999)
            for ghost in ghosts
            if not ghost.eyes_mode
        )

        if self.player.power_pellet_active:
            # move TOWARD ghosts when powered
            reward += (10 / (min_ghost_dist + 1))
        else:
            # move AWAY from ghosts when normal
            if min_ghost_dist < 4:
                reward -= 30 / (min_ghost_dist + 1)  # strong danger zone
            elif min_ghost_dist < 8:
                reward -= 10 / (min_ghost_dist + 1)
            else:
                reward += 0.2   # safe zone small reward


        # --- DOT DISTANCE SHAPING (accelerates learning) ---
        min_dot_dist = float('inf')
        for (x, y), dist in dists.items():
            if map_layout[y][x] == 25:
                min_dot_dist = min(min_dot_dist, dist)

        if min_dot_dist != float('inf'):
            reward += 2.0 / (min_dot_dist + 1)

        # Small step cost (keeps actions efficient)
        reward -= 0.3

        return reward

    def reset(self):
        """Reset the game for RL episode restart."""
        self.reset_game()
        return self.get_state()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # --- RL ADD ---
            if not rl_mode:
                # --- manual control (keyboard) ---
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if (event.key == pygame.K_w or event.key == pygame.K_UP) and not self.game_over:
                            self.player.set_direction("up")
                        elif (event.key == pygame.K_s or event.key == pygame.K_DOWN) and not self.game_over:
                            self.player.set_direction("down")
                        elif (event.key == pygame.K_a or event.key == pygame.K_LEFT) and not self.game_over:
                            self.player.set_direction("left")
                        elif (event.key == pygame.K_d or event.key == pygame.K_RIGHT) and not self.game_over:
                            self.player.set_direction("right")
                        elif (event.key == pygame.K_RETURN and self.lives <= 0) or event.key == pygame.K_ESCAPE:
                            running = False
            else:
                # --- RL mode control ---
                action = self.agent.act(self.get_state())
                if action == 0:
                    self.player.set_direction("up")
                elif action == 1:
                    self.player.set_direction("down")
                elif action == 2:
                    self.player.set_direction("left")
                elif action == 3:
                    self.player.set_direction("right")
                self.state = self.get_state()

            if self.game_over: 
                self.player.score = 0
                running = False
            

            self.screen.fill((0, 0, 0))
            self.map.draw(self.screen)
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 5 and not self.game_over:
                player_x, player_y = self.player.get_position()
                player_direction = self.player.get_direction()
                if self.player.power_pellet_active:
                    for ghost in [self.red_ghost, self.orange_ghost, self.cyan_ghost, self.pink_ghost]:
                        if not ghost.eyes_mode:
                            ghost.image = self.textures["scared_ghost"]
                            ghost.set_target_tile((MAP_WIDTH-player_x,MAP_HEIGHT-player_y))

                # Define custom chase target tiles for each ghost here
                else:
                    self.red_ghost.image = self.textures["red_ghost"]
                    self.pink_ghost.image = self.textures["pink_ghost"]
                    self.cyan_ghost.image = self.textures["cyan_ghost"]
                    self.orange_ghost.image = self.textures["orange_ghost"]
                    for ghost in [self.red_ghost, self.orange_ghost, self.cyan_ghost, self.pink_ghost]:
                        ghost.eyes_mode = False
                    red_target = (player_x, player_y)

                    self.red_ghost.set_target_tile(red_target)
                    pink_x, pink_y = player_x, player_y  # Default to player's current position

                    # Determine pink target based on player direction
                    if player_direction == "up":
                        pink_y -= 4
                        pink_x -= 4  # Move up
                    elif player_direction == "down":
                        pink_y += 4  # Move down
                    elif player_direction == "left":
                        pink_x -= 4  # Move left
                    elif player_direction == "right":
                        pink_x += 4  # Move right

                    pink_target = (pink_x, pink_y)
                    self.pink_ghost.set_target_tile(pink_target)

                    if player_direction == "up":
                        ahead_x, ahead_y = player_x - 2, player_y - 2
                    elif player_direction == "down":
                        ahead_x, ahead_y = player_x, player_y + 2
                    elif player_direction == "left":
                        ahead_x, ahead_y = player_x - 2, player_y
                    elif player_direction == "right":
                        ahead_x, ahead_y = player_x + 2, player_y
                    else:
                        ahead_x, ahead_y = player_x, player_y

                    # Step 2: Get the red ghost's position
                    red_x, red_y = self.red_ghost.get_position()

                    # Step 3: Calculate the vector from the red ghost to the "2 tiles ahead" position
                    vector_x = ahead_x - red_x
                    vector_y = ahead_y - red_y

                    # Step 4: Double the vector to get the cyan ghost's target tile
                    cyan_target_x = red_x + 2 * vector_x
                    cyan_target_y = red_y + 2 * vector_y
                    cyan_target = (cyan_target_x, cyan_target_y)
                    self.cyan_ghost.set_target_tile(cyan_target)

                    orange_pos = self.orange_ghost.get_position()
                    distance_to_player = abs(orange_pos[0] - player_x) + abs(orange_pos[1] - player_y)
                    orange_target = (player_x, player_y) if distance_to_player > 8 else (2, 35)
                    self.orange_ghost.set_target_tile(orange_target)

            # Check for collision with ghosts
            if any(self.player.get_position() == ghost.get_position() for ghost in [self.red_ghost, self.orange_ghost, self.cyan_ghost, self.pink_ghost]) and not self.game_over and not self.victory:
                if self.player.power_pellet_active:
                    for ghost in [self.red_ghost, self.orange_ghost, self.cyan_ghost, self.pink_ghost]:
                        if self.player.get_position() == ghost.get_position() and not ghost.eyes_mode:
                            # sound_manager.play_sound('power_pellet_eat')
                            # Enter eyes mode and set target to (14, 14)
                            self.player.score += 200
                            self.player.just_ate_ghost = True
                            ghost.image = self.textures['ghost_eyes']
                            ghost.set_target_tile((14, 14))
                            ghost.eyes_mode = True  # Set the ghost to eyes mode

                        # Check if the ghost in eyes mode has reached (14, 14)
                        if ghost.eyes_mode and ghost.get_position() == (14, 14):
                            ghost.image = self.textures['scared_ghost']  # Revert to scared ghost texture
                            ghost.eyes_mode = False

                else:
                    self.lives -= 1
                    self.player.just_died = True
                    # sound_manager.play_sound("player_death")
                    if self.lives == 0:
                        map_layout[20][9] = ord('G')
                        map_layout[20][10] = ord('A')
                        map_layout[20][11] = ord('M')
                        map_layout[20][12] = ord('E')
                        map_layout[20][14] = ord('O')
                        map_layout[20][15] = ord('V')
                        map_layout[20][16] = ord('E')
                        map_layout[20][17] = ord('R')
                        map_layout[20][18] = ord('!')
                        self.player.score = 0
                        self.game_over = True
                        # sound_manager.stop_all_sounds()
                        # sound_manager.play_sound("game_over")# End the game
                    else:
                        # Clear tile and reset player position
                        self.reset_player()
                        
                        # Update life counter visuals
                        if self.lives == 1:
                            map_layout[34][0] = 0
                        elif self.lives == 2:
                            map_layout[34][1] = 0

            if self.player.count_dot == 242 and self.player.count_power == 4 and not self.game_over and not self.victory:
                    map_layout[20][10] = ord('V')
                    map_layout[20][11] = ord('I')
                    map_layout[20][12] = ord('C')
                    map_layout[20][13] = ord('T')
                    map_layout[20][14] = ord('O')
                    map_layout[20][15] = ord('R')
                    map_layout[20][16] = ord('Y')
                    map_layout[20][17] = ord('!')
                    self.lives = float('inf')
                    self.victory = True
                    # sound_manager.stop_all_sounds()
                    # sound_manager.play_sound("victory")

            self.all_sprites.update()
            self.all_sprites.draw(self.screen)
            pygame.display.flip()

                    # --- reward collection for RL (if rl_mode) ---
            if rl_mode:
                # get step reward and accumulate; then reset internal self.reward so next step is fresh
                step_reward = self.get_reward()
                # If your get_reward uses self.reward internally, ensure it does not accumulate across steps.
                # Accumulate into per-episode total:
                self.total_reward += step_reward
                # reset any internal reward accumulator to avoid double-counting next frame:
                try:
                    self.reward = 0
                except Exception:
                    pass

                # store and train
                next_state = self.get_state()
                done = self.game_over or self.victory
                self.agent.remember(self.state, action, step_reward, next_state, done)
                self.agent.replay()
                self.step_count += 1
                if self.step_count % 500 == 0:
                    self.agent.update_target()
                if done:
                    # update target network and end episode
                    self.agent.update_target()
                    running = False
                    
                self.clock.tick()

        pygame.quit()
