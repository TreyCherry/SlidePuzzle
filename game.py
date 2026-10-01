import pygame
import sys
import random
from PIL import Image
from prompt_generator import PromptGenerator
from image_generator import ImageGenerator
import os

# Constants
WIDTH, HEIGHT = 680, 550  # Increased width for side menu
GRID_SIZE = 4
TILE_SIZE = 400 // GRID_SIZE
MARGIN = 2
SHUFFLE_MOVES = GRID_SIZE * GRID_SIZE * 10

SAVE_DIRECTORY = "saved_images"

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)


    
class Tile:
    def __init__(self, value, x, y, image):
        self.value = value
        self.x = x
        self.y = y
        self.image = image

    def draw(self, surface):
        if self.value != 0:  # Don't draw the empty tile
            surface.blit(self.image, (self.x * TILE_SIZE, self.y * TILE_SIZE))

class Puzzle:
    def __init__(self, image):
        self.tiles = []
        self.empty_x = GRID_SIZE - 1
        self.empty_y = GRID_SIZE - 1
        self.load_image(image)
        self.initialize()
        self.shuffle()

    def load_image(self, image):
        resized_image = image.resize((400, 400))
        
        self.tile_images = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                box = (x * TILE_SIZE, y * TILE_SIZE, (x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE)
                tile_image = resized_image.crop(box)
                self.tile_images.append(pygame.image.fromstring(tile_image.tobytes(), tile_image.size, tile_image.mode))

    def initialize(self):
        self.tiles = [Tile(i + 1, i % GRID_SIZE, i // GRID_SIZE, self.tile_images[i]) for i in range(GRID_SIZE * GRID_SIZE - 1)]
        self.tiles.append(Tile(0, GRID_SIZE - 1, GRID_SIZE - 1, None))  # Empty tile
        self.empty_x, self.empty_y = GRID_SIZE - 1, GRID_SIZE - 1

    def shuffle(self):
        for _ in range(SHUFFLE_MOVES):
            possible_moves = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = self.empty_x + dx, self.empty_y + dy
                if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                    possible_moves.append((new_x, new_y))
            
            if possible_moves:
                move_x, move_y = random.choice(possible_moves)
                self.move(move_x, move_y)

    def move(self, x, y):
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            if abs(x - self.empty_x) + abs(y - self.empty_y) == 1:
                clicked_index = y * GRID_SIZE + x
                empty_index = self.empty_y * GRID_SIZE + self.empty_x

                self.tiles[clicked_index], self.tiles[empty_index] = self.tiles[empty_index], self.tiles[clicked_index]

                self.tiles[clicked_index].x, self.tiles[clicked_index].y = x, y
                self.tiles[empty_index].x, self.tiles[empty_index].y = self.empty_x, self.empty_y

                self.empty_x, self.empty_y = x, y
                return True
        return False

    def draw(self, surface):
        for tile in self.tiles:
            tile.draw(surface)

    def is_solved(self):
        return all(tile.value == i + 1 for i, tile in enumerate(self.tiles[:-1])) and self.tiles[-1].value == 0

    def complete_image(self):
        self.tiles[-1] = Tile(GRID_SIZE * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1, self.tile_images[-1])
        self.empty_x, self.empty_y = -1, -1  # Move empty tile out of the grid


def draw_button(surface, text, x, y, w, h, font):
    pygame.draw.rect(surface, GRAY, (x, y, w, h))
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=(x + w // 2, y + h // 2))
    surface.blit(text_surface, text_rect)
    return pygame.Rect(x, y, w, h)

def draw_checkbox(surface, text, x, y, checked, font):
    checkbox_size = 20
    pygame.draw.rect(surface, BLACK, (x, y, checkbox_size, checkbox_size), 2)
    if checked:
        pygame.draw.line(surface, BLACK, (x+3, y+10), (x+8, y+15), 2)
        pygame.draw.line(surface, BLACK, (x+8, y+15), (x+17, y+5), 2)
    text_surface = font.render(text, True, BLACK)
    surface.blit(text_surface, (x + checkbox_size + 5, y))
    return pygame.Rect(x, y, checkbox_size + 5 + text_surface.get_width(), checkbox_size)
def main(image_generator, prompt_generator):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Image Sliding Puzzle")
    
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    initial_image = Image.open('initial_image.png')
    puzzle = Puzzle(initial_image)
    clock = pygame.time.Clock()

    themes = prompt_generator.get_available_themes()
    theme_checkboxes = {theme: True for theme in themes}
    
    gallery_mode = False
    gallery_index = 0

    def generate_new_image():
        active_themes = [theme for theme, checked in theme_checkboxes.items() if checked]
        prompt = prompt_generator.generate_prompt(active_themes)
        image_generator.start_generation(prompt, active_themes)

    generate_new_image()  # Generate the first image in the background

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if not gallery_mode:
                    if x < 400 and y < 400:  # Click is within the puzzle area
                        tile_x, tile_y = x // TILE_SIZE, y // TILE_SIZE
                        puzzle.move(tile_x, tile_y)
                    elif reshuffle_button.collidepoint(event.pos):
                        new_image_data = image_generator.get_generated_image()
                        if new_image_data:
                            image_id, new_image = new_image_data
                            puzzle = Puzzle(new_image)
                            generate_new_image()  # Generate next image in the background
                    elif gallery_button.collidepoint(event.pos):
                        gallery_mode = True
                        gallery_index = 0
                    else:
                        for theme, rect in theme_checkbox_rects.items():
                            if rect.collidepoint(event.pos):
                                theme_checkboxes[theme] = not theme_checkboxes[theme]
                                generate_new_image()  # Regenerate image when themes change
                else:
                    if prev_button.collidepoint(event.pos):
                        gallery_index = max(0, gallery_index - 1)
                    elif next_button.collidepoint(event.pos):
                        gallery_index = min(len(image_generator.get_image_history()) - 1, gallery_index + 1)
                    elif exit_gallery_button.collidepoint(event.pos):
                        gallery_mode = False
                    elif use_image_button.collidepoint(event.pos):
                        history = image_generator.get_image_history()
                        if history:
                            _, _, image, _ = history[gallery_index]
                            puzzle = Puzzle(image)
                            gallery_mode = False

        screen.fill(WHITE)

        if not gallery_mode:
            puzzle.draw(screen)

            if puzzle.is_solved():
                puzzle.complete_image()
                puzzle.draw(screen)

            # Draw UI elements
            reshuffle_button = draw_button(screen, "New Puzzle", 10, HEIGHT - 80, 380, 30, font)
            gallery_button = draw_button(screen, "Gallery", 10, HEIGHT - 40, 380, 30, font)
            
            # Draw theme checkboxes
            theme_checkbox_rects = {}
            for i, theme in enumerate(themes):
                checkbox_rect = draw_checkbox(screen, theme, 410, 10 + i*30, theme_checkboxes[theme], small_font)
                theme_checkbox_rects[theme] = checkbox_rect

        else:
            history = image_generator.get_image_history()
            if history:
                image_id, prompt, image, metadata = history[gallery_index]
                screen.blit(pygame.image.fromstring(image.tobytes(), image.size, image.mode), (50, 30))
                prompt_text = small_font.render(f"Prompt: {prompt}", True, BLACK)
                screen.blit(prompt_text, (50, 0))
                # id_text = small_font.render(f"Image ID: {image_id}", True, BLACK)
                # screen.blit(id_text, (50, 20))
                # session_text = small_font.render(f"Session: {metadata['session_id']}", True, BLACK)
                # screen.blit(session_text, (50, HEIGHT - 120))
                timestamp_text = small_font.render(f"Time: {metadata['timestamp']}", True, BLACK)
                screen.blit(timestamp_text, (50, 15))

                prev_button = draw_button(screen, "Previous", 10, HEIGHT - 35, 140, 30, font)
                next_button = draw_button(screen, "Next", 160, HEIGHT - 35, 140, 30, font)
                use_image_button = draw_button(screen, "Use Image", 310, HEIGHT - 35, 140, 30, font)
                exit_gallery_button = draw_button(screen, "Exit Gallery", 460, HEIGHT - 35, 140, 30, font)

        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    image_generator = ImageGenerator(SAVE_DIRECTORY)
    prompt_generator = PromptGenerator("themes")
    main(image_generator, prompt_generator)
    
#Main ver 2    
# def main(image_generator, prompt_generator):
    # pygame.init()
    # screen = pygame.display.set_mode((WIDTH, HEIGHT))
    # pygame.display.set_caption("Image Sliding Puzzle")
    
    # font = pygame.font.Font(None, 36)
    # small_font = pygame.font.Font(None, 24)
    
    # initial_image = Image.open('test.png')
    # puzzle = Puzzle(initial_image)
    # clock = pygame.time.Clock()

    # themes = prompt_generator.get_available_themes()
    # theme_checkboxes = {theme: False for theme in themes}
    
    # gallery_mode = False
    # gallery_index = 0

    # def generate_new_image():
        # active_themes = [theme for theme, checked in theme_checkboxes.items() if checked]
        # prompt = prompt_generator.generate_prompt(active_themes)
        # image_generator.start_generation(prompt, active_themes)

    # generate_new_image()  # Generate the first image in the background

    # while True:
        # for event in pygame.event.get():
            # if event.type == pygame.QUIT:
                # pygame.quit()
                # sys.exit()
            # elif event.type == pygame.MOUSEBUTTONDOWN:
                # x, y = event.pos
                # if not gallery_mode:
                    # if x < 400 and y < 400:  # Click is within the puzzle area
                        # tile_x, tile_y = x // TILE_SIZE, y // TILE_SIZE
                        # puzzle.move(tile_x, tile_y)
                    # elif reshuffle_button.collidepoint(event.pos):
                        # new_image_data = image_generator.get_generated_image()
                        # if new_image_data:
                            # image_id, new_image = new_image_data
                            # puzzle = Puzzle(new_image)
                            # generate_new_image()  # Generate next image in the background
                    # elif gallery_button.collidepoint(event.pos):
                        # gallery_mode = True
                        # gallery_index = 0
                    # else:
                        # for theme, rect in theme_checkbox_rects.items():
                            # if rect.collidepoint(event.pos):
                                # theme_checkboxes[theme] = not theme_checkboxes[theme]
                                # generate_new_image()  # Regenerate image when themes change
                # else:
                    # if prev_button.collidepoint(event.pos):
                        # gallery_index = max(0, gallery_index - 1)
                    # elif next_button.collidepoint(event.pos):
                        # gallery_index = min(len(image_generator.get_image_history()) - 1, gallery_index + 1)
                    # elif exit_gallery_button.collidepoint(event.pos):
                        # gallery_mode = False

        # screen.fill(WHITE)

        # if not gallery_mode:
            # puzzle.draw(screen)

            # if puzzle.is_solved():
                # puzzle.complete_image()
                # puzzle.draw(screen)

            # # Draw UI elements
            # reshuffle_button = draw_button(screen, "New Puzzle", 10, HEIGHT - 80, 380, 30, font)
            # gallery_button = draw_button(screen, "Gallery", 10, HEIGHT - 40, 380, 30, font)
            
            # # Draw theme checkboxes
            # theme_checkbox_rects = {}
            # for i, theme in enumerate(themes):
                # checkbox_rect = draw_checkbox(screen, theme, 410, 10 + i*30, theme_checkboxes[theme], small_font)
                # theme_checkbox_rects[theme] = checkbox_rect

        # else:
            # history = image_generator.get_image_history()
            # if history:
                # image_id, prompt, image = history[gallery_index]
                # screen.blit(pygame.image.fromstring(image.tobytes(), image.size, image.mode), (50, 50))
                # prompt_text = small_font.render(f"Prompt: {prompt}", True, BLACK)
                # screen.blit(prompt_text, (50, 460))
                # id_text = small_font.render(f"Image ID: {image_id}", True, BLACK)
                # screen.blit(id_text, (50, 480))

                # prev_button = draw_button(screen, "Previous", 10, HEIGHT - 40, 180, 30, font)
                # next_button = draw_button(screen, "Next", 200, HEIGHT - 40, 180, 30, font)
                # exit_gallery_button = draw_button(screen, "Exit Gallery", 390, HEIGHT - 40, 180, 30, font)

        # pygame.display.flip()
        # clock.tick(30)

# if __name__ == "__main__":
    # image_generator = ImageGenerator()
    # prompt_generator = PromptGenerator("themes")
    # main(image_generator, prompt_generator)




# import pygame
# import sys
# import random
# from PIL import Image
# from prompt_generator import PromptGenerator
# from image_generator import ImageGenerator

# # Constants
# WIDTH, HEIGHT = 600, 500  # Increased width for side menu
# GRID_SIZE = 4
# TILE_SIZE = 400 // GRID_SIZE
# MARGIN = 2
# SHUFFLE_MOVES = GRID_SIZE * GRID_SIZE * 10

# # Colors
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# GRAY = (200, 200, 200)
# RED = (255, 0, 0)

# class Tile:
    # def __init__(self, value, x, y, image):
        # self.value = value
        # self.x = x
        # self.y = y
        # self.image = image

    # def draw(self, surface):
        # if self.value != 0:  # Don't draw the empty tile
            # surface.blit(self.image, (self.x * TILE_SIZE, self.y * TILE_SIZE))

# class Puzzle:
    # def __init__(self, image):
        # self.tiles = []
        # self.empty_x = GRID_SIZE - 1
        # self.empty_y = GRID_SIZE - 1
        # self.load_image(image)
        # self.initialize()
        # self.shuffle()

    # def load_image(self, image):
        # resized_image = image.resize((400, 400))
        
        # self.tile_images = []
        # for y in range(GRID_SIZE):
            # for x in range(GRID_SIZE):
                # box = (x * TILE_SIZE, y * TILE_SIZE, (x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE)
                # tile_image = resized_image.crop(box)
                # self.tile_images.append(pygame.image.fromstring(tile_image.tobytes(), tile_image.size, tile_image.mode))

    # def initialize(self):
        # self.tiles = [Tile(i + 1, i % GRID_SIZE, i // GRID_SIZE, self.tile_images[i]) for i in range(GRID_SIZE * GRID_SIZE - 1)]
        # self.tiles.append(Tile(0, GRID_SIZE - 1, GRID_SIZE - 1, None))  # Empty tile
        # self.empty_x, self.empty_y = GRID_SIZE - 1, GRID_SIZE - 1

    # def shuffle(self):
        # for _ in range(SHUFFLE_MOVES):
            # possible_moves = []
            # for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                # new_x, new_y = self.empty_x + dx, self.empty_y + dy
                # if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                    # possible_moves.append((new_x, new_y))
            
            # if possible_moves:
                # move_x, move_y = random.choice(possible_moves)
                # self.move(move_x, move_y)

    # def move(self, x, y):
        # if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            # if abs(x - self.empty_x) + abs(y - self.empty_y) == 1:
                # clicked_index = y * GRID_SIZE + x
                # empty_index = self.empty_y * GRID_SIZE + self.empty_x

                # self.tiles[clicked_index], self.tiles[empty_index] = self.tiles[empty_index], self.tiles[clicked_index]

                # self.tiles[clicked_index].x, self.tiles[clicked_index].y = x, y
                # self.tiles[empty_index].x, self.tiles[empty_index].y = self.empty_x, self.empty_y

                # self.empty_x, self.empty_y = x, y
                # return True
        # return False

    # def draw(self, surface):
        # for tile in self.tiles:
            # tile.draw(surface)

    # def is_solved(self):
        # return all(tile.value == i + 1 for i, tile in enumerate(self.tiles[:-1])) and self.tiles[-1].value == 0

    # def complete_image(self):
        # self.tiles[-1] = Tile(GRID_SIZE * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1, self.tile_images[-1])
        # self.empty_x, self.empty_y = -1, -1  # Move empty tile out of the grid

# def draw_button(surface, text, x, y, w, h, font):
    # pygame.draw.rect(surface, GRAY, (x, y, w, h))
    # text_surface = font.render(text, True, BLACK)
    # text_rect = text_surface.get_rect(center=(x + w // 2, y + h // 2))
    # surface.blit(text_surface, text_rect)
    # return pygame.Rect(x, y, w, h)

# def draw_checkbox(surface, text, x, y, checked, font):
    # checkbox_size = 20
    # pygame.draw.rect(surface, BLACK, (x, y, checkbox_size, checkbox_size), 2)
    # if checked:
        # pygame.draw.line(surface, BLACK, (x+3, y+10), (x+8, y+15), 2)
        # pygame.draw.line(surface, BLACK, (x+8, y+15), (x+17, y+5), 2)
    # text_surface = font.render(text, True, BLACK)
    # surface.blit(text_surface, (x + checkbox_size + 5, y))
    # return pygame.Rect(x, y, checkbox_size + 5 + text_surface.get_width(), checkbox_size)

# def main(image_generator, prompt_generator):
    # pygame.init()
    # screen = pygame.display.set_mode((WIDTH, HEIGHT))
    # pygame.display.set_caption("Image Sliding Puzzle")
    
    # font = pygame.font.Font(None, 36)
    # small_font = pygame.font.Font(None, 24)
    
    # initial_image = Image.open('test.png')
    # puzzle = Puzzle(initial_image)
    # clock = pygame.time.Clock()

    # themes = prompt_generator.get_available_themes()
    # theme_checkboxes = {theme: False for theme in themes}
    
    # while True:
        # for event in pygame.event.get():
            # if event.type == pygame.QUIT:
                # pygame.quit()
                # sys.exit()
            # elif event.type == pygame.MOUSEBUTTONDOWN:
                # x, y = event.pos
                # if x < 400 and y < 400:  # Click is within the puzzle area
                    # tile_x, tile_y = x // TILE_SIZE, y // TILE_SIZE
                    # puzzle.move(tile_x, tile_y)
                # elif reshuffle_button.collidepoint(event.pos):
                    # active_themes = [theme for theme, checked in theme_checkboxes.items() if checked]
                    # prompt = prompt_generator.generate_prompt(active_themes)
                    # image_generator.start_generation(prompt)
                # else:
                    # for theme, rect in theme_checkbox_rects.items():
                        # if rect.collidepoint(event.pos):
                            # theme_checkboxes[theme] = not theme_checkboxes[theme]

        # screen.fill(WHITE)
        # puzzle.draw(screen)

        # if puzzle.is_solved():
            # puzzle.complete_image()
            # puzzle.draw(screen)

        # # Draw UI elements
        # reshuffle_button = draw_button(screen, "Regenerate", 10, HEIGHT - 40, 380, 30, font)
        
        # # Draw theme checkboxes
        # theme_checkbox_rects = {}
        # for i, theme in enumerate(themes):
            # checkbox_rect = draw_checkbox(screen, theme, 410, 10 + i*30, theme_checkboxes[theme], small_font)
            # theme_checkbox_rects[theme] = checkbox_rect

        # # Check if a new image is available
        # new_image = image_generator.get_generated_image()
        # if new_image:
            # puzzle = Puzzle(new_image)

        # # Update the button text based on generation status
        # button_text = "Generating..." if image_generator.is_generating() else "Regenerate"
        # reshuffle_button = draw_button(screen, button_text, 10, HEIGHT - 40, 380, 30, font)

        # pygame.display.flip()
        # clock.tick(30)

# if __name__ == "__main__":
    # image_generator = ImageGenerator()
    # prompt_generator = PromptGenerator("themes")
    # main(image_generator, prompt_generator)
