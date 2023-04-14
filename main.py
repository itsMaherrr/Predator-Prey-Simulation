import sys

import time
import pygame
import random
import threading
import pycuda.driver as cuda
import pycuda.autoinit
from prey import Prey
from predator import Predator
from gpu_functions import mod

import numpy as np

x = 0
y = 1
keep_checking = True

window_width, window_height = 1920, 1080
game_width, game_height = 960, 863
screen_bg_color = (160, 160, 160, 90)
window_bg_color = (128, 128, 128)
red = (255, 0, 0)
neon_green = (15, 255, 80)
angles = [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12]
clock = pygame.time.Clock()

radius = 1
velocity = 2

predators = [Predator(red, random.randint(radius, game_width - radius), i, random.randint(radius, game_height - radius),
                      brain=np.array([np.random.rand(10), np.random.rand(10)]), radius=radius, velocity=velocity) for i in range(200)]
preys = [Prey(neon_green, random.randint(radius, game_width - radius), i, random.randint(radius, game_height - radius),
              brain=np.array([np.random.rand(10), np.random.rand(10)]), radius=radius, velocity=velocity) for i in range(1000)]

def get_collision(entities1, entities2):
    if entities1 and entities2:

        surviving_preys = mod.get_function("surviving_preys")

        actual_preys = np.array([actual_prey.position for actual_prey in entities1]).astype(np.float32).flatten()
        actual_predators = np.array([actual_predator.position for actual_predator in entities2])\
            .astype(np.float32).flatten()
        fed_predators = np.zeros(int(len(predators) / 2)).astype(np.int32)
        alive_preys = np.ones(int(len(actual_preys) / 2))

        actual_preys_gpu = cuda.mem_alloc(actual_preys.nbytes)
        actual_predators_gpu = cuda.mem_alloc(actual_predators.nbytes)
        fed_predators_gpu = cuda.mem_alloc(fed_predators.nbytes)
        alive_preys_gpu = cuda.mem_alloc(alive_preys.nbytes)

        cuda.memcpy_htod(actual_preys_gpu, actual_preys)
        cuda.memcpy_htod(actual_predators_gpu, actual_predators)
        cuda.memcpy_htod(fed_predators_gpu, fed_predators)
        cuda.memcpy_htod(alive_preys_gpu, alive_preys)

        block_size = int(len(actual_predators) / 2)
        num_blocks = int(len(actual_preys) / 2)
        grid_size = (num_blocks, 1, 1)
        block_size = (block_size, 1, 1)

        surviving_preys(actual_preys_gpu, actual_predators_gpu, fed_predators_gpu, alive_preys_gpu, np.int32(radius),
                        block=block_size, grid=grid_size)

        collided_entities = np.empty_like(alive_preys)
        fed_preds = np.empty_like(fed_predators)
        cuda.memcpy_dtoh(collided_entities, alive_preys_gpu)
        cuda.memcpy_dtoh(fed_preds, fed_predators_gpu)

        return np.array(collided_entities), np.array(fed_preds)
    return np.array([]), np.array([])

def reproduct_preys():
    collided_preys = get_collision(preys, preys)


def reproduct_predators():
    collided_predators = get_collision(predators, predators)

def update_preys():
    eaten_preys, fed_predators = get_collision(preys, predators)
    j = 0
    for i in range(len(eaten_preys)):
        if eaten_preys[i] == 0:
            del preys[i - j]
            j += 1
    for i in np.where(fed_predators > 0)[0]:
        predators[i].eat_prey(fed_predators[i])

def calculate_distances(entities_1, entities_2):
    return np.linalg.norm(entities_1[:, np.newaxis] - entities_2, axis=2)

def get_all_distances():
    if len(preys) != 0 and len(predators) != 0:
        actual_preys = np.array([actual_prey.position for actual_prey in preys])
        actual_predators = np.array([actual_predator.position for actual_predator in predators])

        prey_pred_distance = calculate_distances(actual_preys, actual_predators)
        prey_prey_distance = calculate_distances(actual_preys, actual_preys)
        pred_pred_distance = calculate_distances(actual_predators, actual_predators)

        return np.array([prey_pred_distance, prey_prey_distance, pred_pred_distance])

def update_surviving_preys(distances ,radius):
    index_preys = 0
    index_predators = 1

    all_distances = get_all_distances()

    collided = np.where(all_distances <= 2 * radius)

    j = 0
    for i in range(len(collided[index_preys])):
        x = collided[index_preys][i]
        del preys[x - j]
        j += 1

def init_all_images():
    prey_img = pygame.transform.smoothscale(pygame.image.load("assets/500px_green.png").convert_alpha(), (5, 2))
    predator_img = pygame.transform.smoothscale(pygame.image.load("assets/500px_red.png").convert_alpha(), (5, 2))

    prey_img_angles = [pygame.transform.rotate(prey_img, i).convert_alpha() for i in range(360)]
    predator_img_angles = [pygame.transform.rotate(predator_img, i).convert_alpha() for i in range(360)]

    return prey_img_angles, predator_img_angles


if __name__ == '__main__':

    pygame.init()

    window = pygame.display.set_mode((window_width, window_height), pygame.FULLSCREEN)
    pygame.display.set_caption("Lotka-Volterra")
    screen = pygame.Surface((game_width, game_height))
    mini_map = pygame.Surface((game_width // 4, game_height // 4))
    prey_img_angles, predator_img_angles = init_all_images()
    running = True

    while running:
        update_preys()
        #survival_checking_thread = threading.Thread(target=reproduct_predators())
        #survival_checking_thread.start()
        #survival_checking_thread = threading.Thread(target=reproduct_preys())
        #survival_checking_thread.start()
        #distances = get_all_distances()
        window.fill(window_bg_color)
        screen.fill(screen_bg_color)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                for predator in predators:
                    predator.stop_counting()
                for prey in preys:
                    prey.stop_counting()

        for predator in predators:
            predator.move()
            predator_shape = predator_img_angles[predator.get_view_angle_in_degrees()]
            predator.draw_view_range(screen, angles, predator_shape.get_rect().center)

            if predator.position[x] <= radius or predator.position[x] >= game_width - radius:
                predator.rebound()
            if predator.position[y] <= radius or predator.position[y] >= game_height - radius:
                predator.rebound()

            screen.blit(predator_shape, predator.position)

        for prey in preys:
            prey.move()
            prey_shape = prey_img_angles[prey.get_view_angle_in_degrees()]
            prey.draw_view_range(screen, angles, prey_shape.get_rect().center)
            if prey.position[x] <= radius or prey.position[x] >= game_width - radius:
                prey.rebound()
            if prey.position[y] <= radius or prey.position[y] >= game_height - radius:
                prey.rebound()

            """
            for predator in predators:
                distance = ((prey.position[x] - predator.position[x]) ** 2 + (prey.position[y] - predator.position[y]) ** 2) ** 0.5
                if distance <= 2 * radius:
                    collided = True
                    break
            """
            screen.blit(prey_shape, prey.position)
            # mini_map.blit(screen, ())

        window.blit(screen, (window_width / 16, 0))
        print(clock.get_fps())
        pygame.display.update()
        clock.tick(60)

        pygame.display.flip()
        #pygame.time.delay(10)

    pygame.quit()