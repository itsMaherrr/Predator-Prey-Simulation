import sys

import time
import pygame
import pygame.math as pgm
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

window_width, window_height = 1920 // 2, 1080 // 2
game_width, game_height = 930 // 2, 833 // 2
minimap_width, minimap_height = game_width // 4, game_height // 4
screen_bg_color = (96, 96, 96)
window_bg_color = (128, 128, 128)
red = (255, 0, 0)
neon_green = (15, 255, 80)
rect_width = 1
predator_view_angles = np.linspace(-23, 23, num=24) % 360
prey_view_angles = np.linspace(-127, 126, num=24) % 360
clock = pygame.time.Clock()

radius = 1
velocity = 2

predators_num = 100
preys_num = 700

continue_counting = True

predators = [Predator(red, random.randint(radius, game_width - radius), i, random.randint(radius, game_height - radius),
                      brain=np.array([np.random.rand(48), np.random.rand(48)]), radius=radius, velocity=velocity) for i in range(predators_num)]
preys = [Prey(neon_green, random.randint(radius, game_width - radius), i, random.randint(radius, game_height - radius),
              brain=np.array([np.random.rand(48), np.random.rand(48)]), radius=radius, velocity=velocity) for i in range(preys_num)]

def get_collision(entities1, entities2):
    if entities1 and entities2:

        surviving_preys = mod.get_function("surviving_preys")

        actual_preys = np.array([actual_prey.position for actual_prey in entities1]).astype(np.float32).flatten()
        actual_predators = np.array([actual_predator.position for actual_predator in entities2])\
            .astype(np.float32).flatten()
        fed_predators = np.zeros(int(len(actual_predators) / 2)).astype(np.int32)
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

def get_vision(predators, preys):
    if predators and preys:
        find_angles = mod.get_function("find_angles")
        actual_predators = np.array([(predator.position[0], predator.position[1], predator.get_view_angle()) for predator in predators]).flatten().astype(np.float32)
        actual_entities = np.concatenate(
            (np.array([prey.position for prey in preys]), np.array([predator.position for predator in predators]))).flatten().astype(np.float32)
        nbr_predators = len(predators)
        nbr_preys = len(preys)
        nbr_entities = nbr_preys + nbr_predators
        result = np.ones(nbr_predators * nbr_entities * 2).astype(np.float32)

        predators_gpu = cuda.mem_alloc(actual_predators.nbytes)
        entities_gpu = cuda.mem_alloc(actual_entities.nbytes)
        result_gpu = cuda.mem_alloc(result.nbytes)

        cuda.memcpy_htod(predators_gpu, actual_predators)
        cuda.memcpy_htod(entities_gpu, actual_entities)
        cuda.memcpy_htod(result_gpu, result)

        block_size = 1024
        marge = nbr_entities - block_size
        num_blocks = nbr_predators + (marge * nbr_predators + block_size - 1) // block_size
        grid_size = (num_blocks, 1, 1)
        block_size = (block_size, 1, 1)

        find_angles(predators_gpu, entities_gpu, result_gpu, np.int32(nbr_predators), np.int32(nbr_entities),
                    block=block_size,
                    grid=grid_size)

        final_result = np.empty_like(result)
        cuda.memcpy_dtoh(final_result, result_gpu)
        final_result = final_result.reshape((nbr_predators, nbr_entities, 2))

        #indexes = np.where(np.isin(final_result[:, :, 0], predator_view_angles))
        return final_result
    return np.array([])


def make_acts(entities):
    while continue_counting:
        for entity in entities:
            entity.act()
        time.sleep(1/30)


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

def handle_vision(predators, preys, view_angles):
    vision = get_vision(predators, preys)
    enemies_nbr = len(preys)
    for i in range(vision.shape[0]):
        #handle_entities_thread = threading.Thread(target=predators[i].handle_nearby_entities, args=(vision[i], enemies_nbr, view_angles))
        #handle_entities_thread.start()
        #predators[i].handle_nearby_entities(vision[i], enemies_nbr, view_angles)
        predators[i].set_nearby_objects(vision[i])

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
    prey_img = pygame.transform.smoothscale(pygame.image.load("assets/500px_green.png").convert_alpha(), (4, 2))
    predator_img = pygame.transform.smoothscale(pygame.image.load("assets/500px_red.png").convert_alpha(), (4, 2))

    prey_img_angles = [pygame.transform.rotate(prey_img, i).convert_alpha() for i in range(360)]
    predator_img_angles = [pygame.transform.rotate(predator_img, i).convert_alpha() for i in range(360)]

    return prey_img_angles, predator_img_angles


if __name__ == '__main__':

    pygame.init()

    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Lotka-Volterra")
    screen = pygame.Surface((game_width, game_height))
    mini_map = pygame.Surface((minimap_width, minimap_height))
    prey_img_angles, predator_img_angles = init_all_images()

    predators_acting_thread = threading.Thread(target=make_acts, args=(predators,))
    predators_acting_thread.start()
    preys_acting_thread = threading.Thread(target=make_acts, args=(preys,))
    preys_acting_thread.start()

    running = True

    while running:
        update_preys()
        handle_vision(predators, preys, predator_view_angles)
        handle_vision(preys, predators, prey_view_angles)
        #distances = get_all_distances()
        window.fill(window_bg_color)
        screen.fill(screen_bg_color)
        mini_map.fill((240, 240, 240))
        mini_map.set_alpha(80)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                for predator in predators:
                    predator.stop_counting()
                for prey in preys:
                    prey.stop_counting()
            if event.type == pygame.MOUSEBUTTONUP:
                position = pygame.mouse.get_pos()
                # screen.blit(pygame.transform.scale_by(screen, 1.1), (0, 0),
                #                 position + ([window_height * 2] * 2))


        for predator in predators:
            predator.move()
            predator_shape = predator_img_angles[predator.get_view_angle_in_degrees()]
            predator.draw_view_range(screen, predator_view_angles, predator_shape.get_rect().center)
            """
            direction = pgm.Vector2(1, 0).rotate(-predator.get_view_angle_in_degrees())
            line_of_vision = direction * predator.view_range
            for prey in preys:
                vec_to_other = np.array(prey.position) - np.array(predator.position)
                angle = line_of_vision.angle_to(vec_to_other)
                if int(angle % 360) in predator_view_angles:
                    distance = np.linalg.norm(vec_to_other)
                    if distance <= predator.view_range + prey.radius:
                        print(f"predator {predator.id} sees prey {prey.id}")
            """
            """
            if predator.position[x] <= radius or predator.position[x] >= game_width - radius:
                predator.rebound()
            if predator.position[y] <= radius or predator.position[y] >= game_height - radius:
                predator.rebound()
            """
            screen.blit(predator_shape, predator.position)
            mini_map.blit(predator_shape, np.array(predator.position)/4)

        for prey in preys:
            prey.move()
            prey_shape = prey_img_angles[prey.get_view_angle_in_degrees()]
            prey.draw_view_range(screen, prey_view_angles, prey_shape.get_rect().center)
            """
            if prey.position[x] <= radius or prey.position[x] >= game_width - radius:
                prey.rebound()
            if prey.position[y] <= radius or prey.position[y] >= game_height - radius:
                prey.rebound()
            """
            """
            for predator in predators:
                distance = ((prey.position[x] - predator.position[x]) ** 2 + (prey.position[y] - predator.position[y]) ** 2) ** 0.5
                if distance <= 2 * radius:
                    collided = True
                    break
            """
            screen.blit(prey_shape, prey.position)
            mini_map.blit(prey_shape, np.array(prey.position) / 4)

        pygame.draw.rect(window, (255, 255, 255), (window_width / 1.5 - rect_width, 15 - rect_width, minimap_width + 2 * rect_width, minimap_height + 2 * rect_width), width=rect_width)
        #mini_map.blit(resized_screen, (0, 0))
        window.blit(screen, (window_width / 16, 15))
        window.blit(mini_map, (window_width / 1.5, 15))
        print(clock.get_fps())
        pygame.display.update()
        clock.tick(60)

        pygame.display.flip()
        #pygame.time.delay(10)

    pygame.quit()