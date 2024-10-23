import pygame
import random
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import cv2

from settings import *


class Player(pygame.sprite.Sprite):
    def __init__(self, position, size):
        super().__init__()
        self.image = pygame.Surface(size)
        self.image.fill(WHITE)
        self.rect = self.image.get_rect(center=position)
    
    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def update(self, y_pos):
        self.rect.centery = y_pos


class Obstacle(pygame.sprite.Sprite):
    def __init__(self, position, size):
        super().__init__()
        self.image = pygame.Surface(size)
        self.image.fill(WHITE)
        self.rect = self.image.get_rect(topleft=position)
        self.speed = 5

    def update(self):
        self.rect.x -= self.speed
        if self.rect.right < 0:
            self.kill()


class ScoreBox(pygame.sprite.Sprite):
    def __init__(self, position, size):
        super().__init__()
        self.image = pygame.Surface(size)
        self.rect = self.image.get_rect(topleft=position)
        self.speed = 5

    def update(self):
        self.rect.x -= self.speed


def draw_landmarks(image, pose_landmarks):
    h, w, _ = image.shape
    start_point = (int(pose_landmarks[LEFT_SHOULDER].x * w), 
                   int(pose_landmarks[LEFT_SHOULDER].y * h))
    end_point = (int(pose_landmarks[RIGHT_SHOULDER].x * w), 
                 int(pose_landmarks[RIGHT_SHOULDER].y * h))
    cv2.line(image, start_point, end_point, (0, 0, 255), 5)


def normalize_y_pos(y_pos):
    if y_pos < UPPER_LIMIT:
        return 0
    elif y_pos > LOWER_LIMIT:
        return 1

    return (y_pos - UPPER_LIMIT) / (LOWER_LIMIT - UPPER_LIMIT)


def init_game():
    global current_score
    current_score = 0
    
    for obstacle in obstacles:
        obstacle.kill()

    for score_box in score_boxes:
        score_box.kill()


def draw_game_screen():
    player.update(normalized_y_pos * GAME_WINDOW_HEIGHT)
    player.draw(screen)

    score_boxes.update()
    obstacles.update()
    obstacles.draw(screen)

    current_score_text = game_font.render(f"Your Score: {current_score}", True, WHITE)
    screen.blit(current_score_text, (20, 20))

    best_score_text = game_font.render(f"Best Score: {best_score}", True, WHITE)
    screen.blit(best_score_text, (20, 50))


def draw_start_screen():
    font = pygame.font.Font(None, 80)
    text = font.render("Press any key to start", True, WHITE)
    text_rect = text.get_rect(center=(GAME_WINDOW_WIDTH // 2, GAME_WINDOW_HEIGHT // 2))
    screen.blit(text, text_rect)


cap = cv2.VideoCapture(0)
options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

pygame.init()
screen = pygame.display.set_mode((GAME_WINDOW_WIDTH, GAME_WINDOW_HEIGHT))
pygame.display.set_caption("Motion Game")
clock = pygame.time.Clock()
game_font = pygame.font.Font(None, 24)

best_score = 0
current_score = 0

player = Player(position=(100, GAME_WINDOW_HEIGHT // 2), size=(50, 50))
score_boxes = pygame.sprite.Group()
obstacles = pygame.sprite.Group()
obstacle_spawn_time = pygame.time.get_ticks()

running = True
start = False

while running:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if not start:
                start = True
                init_game()

    ret, frame = cap.read()
    if not ret:
        running = False

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (CAMERA_WINDOW_WIDTH, CAMERA_WINDOW_HEIGHT))
    frame = frame[:, CAMERA_WINDOW_MARGIN:CAMERA_WINDOW_WIDTH-CAMERA_WINDOW_MARGIN]

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        draw_landmarks(frame, result.pose_landmarks[0])
        shoulder_y_pos = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
        normalized_y_pos = normalize_y_pos(shoulder_y_pos * CAMERA_WINDOW_HEIGHT)

    cv2.line(frame, (0, UPPER_LIMIT), (CAMERA_WINDOW_WIDTH, UPPER_LIMIT), (0, 0, 255), 5)
    cv2.line(frame, (0, LOWER_LIMIT), (CAMERA_WINDOW_WIDTH, LOWER_LIMIT), (0, 0, 255), 5)
    cv2.imshow("Pose Camera", frame)

    screen.fill(BLACK)

    if start:
        if pygame.time.get_ticks() - obstacle_spawn_time >= OBSTACLE_COOLDOWN:
            space_y_pos = random.randint(0, GAME_WINDOW_HEIGHT - SPACE_HEIGHT)
            score_box = ScoreBox(position=(GAME_WINDOW_WIDTH, space_y_pos), size=(40, SPACE_HEIGHT))
            score_boxes.add(score_box)

            upper_obstacle = Obstacle(position=(GAME_WINDOW_WIDTH, 0), size=(40, space_y_pos))
            lower_obstacle = Obstacle(position=(GAME_WINDOW_WIDTH, space_y_pos + SPACE_HEIGHT), 
                                        size=(40, GAME_WINDOW_HEIGHT - space_y_pos - SPACE_HEIGHT))
            obstacles.add(upper_obstacle)
            obstacles.add(lower_obstacle)
            
            obstacle_spawn_time = pygame.time.get_ticks()

        if pygame.sprite.spritecollide(player, obstacles, False):
            current_score = 0
            start = False

        if pygame.sprite.spritecollide(player, score_boxes, True):
            current_score += 1
            best_score = max(current_score, best_score)

        draw_game_screen()
    else:
        draw_start_screen()
    
    pygame.display.update()
    

cap.release()
cv2.destroyAllWindows()
