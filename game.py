import pygame
from settings import *
from map import *
import time
from player import Player
from ghost import Ghost

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pacman")
        self.clock = pygame.time.Clock()
        
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
        }
        
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

        # Reset ghosts with their initial positions
        self.red_ghost = Ghost(17, 12, self.textures['red_ghost'], (25, 0))
        self.orange_ghost = Ghost(12, 12, self.textures['orange_ghost'], (2, 35))
        self.cyan_ghost = Ghost(17, 17, self.textures['cyan_ghost'], (26, 36))
        self.pink_ghost = Ghost(12, 12, self.textures['pink_ghost'], (2, 0))

        self.all_sprites = pygame.sprite.Group(self.player, self.red_ghost, self.orange_ghost, self.cyan_ghost, self.pink_ghost)

        self.start_time = time.time()
        self.lives = 3  # Reset lives
        self.player.score = 0 

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w or event.key == pygame.K_UP:
                        self.player.set_direction("up")
                    elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                        self.player.set_direction("down")
                    elif event.key == pygame.K_a or event.key == pygame.K_LEFT:
                        self.player.set_direction("left")
                    elif event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                        self.player.set_direction("right")
                    elif (event.key == pygame.K_RETURN and self.lives <= 0) or event.key == pygame.K_ESCAPE:
                        running = False
            self.screen.fill((0, 0, 0))
            self.map.draw(self.screen)

            elapsed_time = time.time() - self.start_time
            if elapsed_time > 5:
                # Define custom chase target tiles for each ghost here
                player_x, player_y = self.player.get_position()
                player_direction = self.player.get_direction()

                red_target = (player_x, player_y)

                self.red_ghost.set_target_tile(red_target)
                pink_x, pink_y = player_x, player_y  # Default to player's current position

                # Determine pink target based on player direction
                if player_direction == "up":
                    pink_y -= 4  # Move up
                elif player_direction == "down":
                    pink_y += 4  # Move down
                elif player_direction == "left":
                    pink_x -= 4  # Move left
                elif player_direction == "right":
                    pink_x += 4  # Move right

                pink_target = (pink_x, pink_y)
                self.pink_ghost.set_target_tile(pink_target)

                if player_direction == "up":
                    ahead_x, ahead_y = player_x, player_y - 2
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
            if any(self.player.get_position() == ghost.get_position() for ghost in [self.red_ghost, self.orange_ghost, self.cyan_ghost, self.pink_ghost]):
                self.lives -= 1
                if self.lives == 0:
                    map_layout[20][9] = ord('G')
                    map_layout[20][10] = ord('A')
                    map_layout[20][11] = ord('M')
                    map_layout[20][12] = ord('E')
                    map_layout[20][14] = ord('O')
                    map_layout[20][15] = ord('V')
                    map_layout[20][16] = ord('E')
                    map_layout[20][17] = ord('R')
                    map_layout[20][18] = ord('!')# End the game
                else:
                    # Clear tile and reset player position
                    self.reset_player()
                    
                    # Update life counter visuals
                    if self.lives == 1:
                        map_layout[34][0] = 0
                    elif self.lives == 2:
                        map_layout[34][1] = 0

            self.all_sprites.update()
            self.all_sprites.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
