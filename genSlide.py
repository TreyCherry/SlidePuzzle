import pygame
import sys
import random
from PIL import Image
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 400, 500  # Increased height for UI elements
GRID_SIZE = 4
TILE_SIZE = WIDTH // GRID_SIZE
MARGIN = 2
SHUFFLE_MOVES = GRID_SIZE * GRID_SIZE * 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Image Sliding Puzzle")

# Fonts
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)

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
    def __init__(self, image_path):
        self.tiles = []
        self.empty_x = GRID_SIZE - 1
        self.empty_y = GRID_SIZE - 1
        self.load_image(image_path)
        self.initialize()
        self.shuffle()

    def load_image(self, image_path):
        original_image = Image.open(image_path)
        resized_image = original_image.resize((WIDTH, WIDTH))
        
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

def draw_button(surface, text, x, y, w, h):
    pygame.draw.rect(surface, GRAY, (x, y, w, h))
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=(x + w // 2, y + h // 2))
    surface.blit(text_surface, text_rect)
    return pygame.Rect(x, y, w, h)

def draw_textbox(surface, text, x, y, w, h):
    pygame.draw.rect(surface, WHITE, (x, y, w, h))
    pygame.draw.rect(surface, BLACK, (x, y, w, h), 2)
    text_surface = small_font.render(text, True, BLACK)
    surface.blit(text_surface, (x + 5, y + 5))
    return pygame.Rect(x, y, w, h)

def draw_loading_bar(surface, progress):
    pygame.draw.rect(surface, WHITE, (50, HEIGHT // 2 - 20, WIDTH - 100, 40))
    pygame.draw.rect(surface, BLACK, (50, HEIGHT // 2 - 20, WIDTH - 100, 40), 2)
    inner_width = (WIDTH - 104) * progress
    pygame.draw.rect(surface, GRAY, (52, HEIGHT // 2 - 18, inner_width, 36))
    pygame.display.flip()

def generate_image(prompt, model_path):
    # Load the diffusion model from the provided path
    pipe = DiffusionPipeline.from_pretrained("stabilityai/sd-turbo")
    #pipe = StableDiffusionPipeline.from_pretrained(model_path)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate an image based on the prompt
    with torch.autocast("cuda"):
        image = pipe(prompt=prompt,num_inference_steps=1, guidance_scale=0.0).images[0]
    
    # Save the image locally
    image_path = "generated_image.png"
    image.save(image_path)
    return image_path

def main():
    model_path = "path_to_your_model_directory"  # Update this to the actual path of your model
    puzzle = Puzzle('test.png')
    clock = pygame.time.Clock()
    prompt = "Sci-fi photo, robot working on laptop"
    textbox_rect = pygame.Rect(10, HEIGHT - 40, WIDTH - 120, 30)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if y < WIDTH:  # Click is within the puzzle area
                    tile_x, tile_y = x // TILE_SIZE, y // TILE_SIZE
                    puzzle.move(tile_x, tile_y)
                elif reshuffle_button.collidepoint(event.pos):
                    new_image_path = generate_image(prompt, model_path)
                    puzzle = Puzzle(new_image_path)
            elif event.type == pygame.KEYDOWN:
                if textbox_rect.collidepoint(pygame.mouse.get_pos()):
                    if event.key == pygame.K_BACKSPACE:
                        prompt = prompt[:-1]
                    else:
                        prompt += event.unicode

        screen.fill(WHITE)
        puzzle.draw(screen)

        if puzzle.is_solved():
            puzzle.complete_image()
            puzzle.draw(screen)

        # Draw UI elements
        textbox_rect = draw_textbox(screen, prompt, 5, HEIGHT - 70, WIDTH - 10, 30)
        reshuffle_button = draw_button(screen, "Regenerate", 10, HEIGHT - 40, WIDTH - 20, 30)
        
        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()




