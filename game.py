import pygame
from settings import *
from map import *
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
        
        self.map = Map()
        self.player = Player(13.5, 26, self.textures)
        
        # Define ghosts with their scatter target tiles
        self.ghosts = pygame.sprite.Group(
            Ghost(17, 12, self.textures['red_ghost'], (25, 0)),
            Ghost(17, 12, self.textures['orange_ghost'], (0, 35)),
            Ghost(17, 12, self.textures['cyan_ghost'], (27, 35)),
            Ghost(17, 12, self.textures['pink_ghost'], (2, 0))
        )
        
        self.all_sprites = pygame.sprite.Group(self.player, *self.ghosts)

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

            self.screen.fill((0, 0, 0))
            self.map.draw(self.screen)
            
            for ghost in self.ghosts:
                ghost.scatter_mode_update(map_layout)
            
            self.all_sprites.update()
            self.all_sprites.draw(self.screen)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
