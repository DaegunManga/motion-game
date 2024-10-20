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


def draw_landmarks(image, pose_landmarks, pose_connections):
    h, w, _ = image.shape

    for start_idx, end_idx in pose_connections:
        start_point = (int(pose_landmarks[start_idx].x * w), 
                       int(pose_landmarks[start_idx].y * h))
        end_point = (int(pose_landmarks[end_idx].x * w), 
                     int(pose_landmarks[end_idx].y * h))
        cv2.line(image, start_point, end_point, (255, 255, 255), 3)

    for landmark in pose_landmarks:
        cx = int(landmark.x * w)
        cy = int(landmark.y * h)
        cv2.circle(image, (cx, cy), 7, (0, 165, 255), cv2.FILLED)
        cv2.circle(image, (cx, cy), 7, (255, 255, 525), 2)


def init_game():
    global start_time
    start_time = pygame.time.get_ticks()
    
    for obstacle in obstacles:
        obstacle.kill()


def draw_game_screen():
    player.update(shoulder_y_pos * SCREEN_HEIGHT)
    player.draw(screen)

    obstacles.update()
    obstacles.draw(screen)

    current_score_text = game_font.render(f"Your Score: {current_score}", True, WHITE)
    screen.blit(current_score_text, (20, 20))

    best_score_text = game_font.render(f"Best Score: {best_score}", True, WHITE)
    screen.blit(best_score_text, (20, 50))


def draw_start_screen():
    font = pygame.font.Font(None, 80)
    text = font.render("Press any key to start", True, WHITE)
    text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    screen.blit(text, text_rect)


cap = cv2.VideoCapture(0)
options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Motion Game")
clock = pygame.time.Clock()
game_font = pygame.font.Font(None, 24)

best_score = 0
current_score = 0

player = Player(position=(100, SCREEN_HEIGHT // 2), size=(50, 50))
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

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

    if result.pose_landmarks:
        for landmarks in result.pose_landmarks:
            draw_landmarks(frame, landmarks, POSE_CONNECTIONS)
            shoulder_y_pos = (landmarks[11].y + landmarks[12].y) / 2

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (800, 450), cv2.INTER_LINEAR)
    cv2.imshow("Pose Camera", frame)

    screen.fill(BLACK)

    if start:
        if pygame.time.get_ticks() - obstacle_spawn_time >= OBSTACLE_COOLDOWN:
            space_y_pos = random.randint(0, SCREEN_HEIGHT - SPACE_HEIGHT)
            upper_obstacle = Obstacle(position=(SCREEN_WIDTH, 0), size=(40, space_y_pos))
            lower_obstacle = Obstacle(position=(SCREEN_WIDTH, space_y_pos + SPACE_HEIGHT), 
                                        size=(40, SCREEN_HEIGHT - space_y_pos - SPACE_HEIGHT))
            obstacles.add(upper_obstacle)
            obstacles.add(lower_obstacle)
            
            obstacle_spawn_time = pygame.time.get_ticks()

        if pygame.sprite.spritecollide(player, obstacles, False):
            current_score = 0
            start = False

        current_score = (pygame.time.get_ticks() - start_time) // 100
        best_score = max(current_score, best_score)

        draw_game_screen()
    else:
        draw_start_screen()
    
    pygame.display.update()
    

cap.release()
cv2.destroyAllWindows()
