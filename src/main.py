import os.path
import sys

import time
import pygame
import random
import threading
import pycuda.autoinit
import pycuda.driver as cuda
import cProfile
import pstats
import pickle
from prey import Prey
from predator import Predator
from kernel_functions import mod
from genetic_algorithm import Evolution
from data_structures import CustomStack

import numpy as np

x = 0
y = 1
keep_checking = True

alpha = 90

window_width, window_height = 1920, 1080
game_width, game_height = 930, 833
minimap_width, minimap_height = game_width // 4, game_height // 4
screen_bg_color = (72, 72, 72)
window_bg_color = (128, 128, 128)
red = (255, 0, 0)
neon_green = (15, 255, 80)
rect_width = 1
predator_view_angles = np.linspace(-23, 23, num=24) % 360
prey_view_angles = np.linspace(-105, 102, num=24) % 360
clock = pygame.time.Clock()

evolution_system = Evolution()

predators_stats = CustomStack(455)
preys_stats = CustomStack(455)

radius = 2
velocity = 2

predators_num = 750
preys_num = 910

max_predators = 674
max_preys = 674

input_size = (2, 48)
bias_size = 2

continue_counting = True

lock = threading.Lock()

predators_brains_nbr, preys_brains_nbr, generation = 0, 0, 0

sys.path.append('assets')
sys.path.append('data')
sys.path.append('stats')

if os.path.getsize("data/saving") > 0:
    with open("data/saving", "rb") as f:
        saved = pickle.load(f)
        predators_brains = saved.get('predators_brains')
        preys_brains = saved.get('preys_brains')
        generation = int(saved.get('generation'))
else:
    predators_brains = [[np.random.uniform(-1, 1, size=input_size), np.random.uniform(-1, 1, bias_size)] for _ in range(predators_num)]
    preys_brains = [[np.random.uniform(-1, 1, size=input_size), np.random.uniform(-1, 1, bias_size)] for _ in range(preys_num)]

predators_brains_nbr = len(predators_brains)
preys_brains_nbr = len(preys_brains)

predators = [Predator(red, [random.randint(radius, game_width - radius), random.randint(radius, game_height - radius)], i,
                      brain=predators_brains[(i % predators_brains_nbr)], radius=radius, velocity=velocity) for i
             in range(max(predators_num, predators_brains_nbr))]
preys = [Prey(neon_green, [random.randint(radius, game_width - radius), random.randint(radius, game_height - radius)], i,
              brain=preys_brains[(i % preys_brains_nbr)], radius=radius, velocity=velocity) for i in
         range(max(preys_num, preys_brains_nbr))]

dead_predators = []

last_gen_predators = []
last_gen_preys = []


def get_collision(entities1, entities2):
    if entities1 and entities2:
        surviving_preys = mod.get_function("surviving_preys")

        predators_num = len(predators)
        preys_num = len(preys)

        actual_preys = np.array([actual_prey.position for actual_prey in entities1]).astype(np.float32).flatten()
        actual_predators = np.array([actual_predator.position for actual_predator in entities2]) \
            .astype(np.float32).flatten()
        collision = np.zeros(predators_num * preys_num).astype(np.int64)

        actual_preys_gpu = cuda.mem_alloc(actual_preys.nbytes)
        actual_predators_gpu = cuda.mem_alloc(actual_predators.nbytes)
        collision_gpu = cuda.mem_alloc(collision.nbytes)

        cuda.memcpy_htod(actual_preys_gpu, actual_preys)
        cuda.memcpy_htod(actual_predators_gpu, actual_predators)
        cuda.memcpy_htod(collision_gpu, collision)

        block_size = int(predators_num)
        num_blocks = int(preys_num)
        grid_size = (num_blocks, 1, 1)
        block_size = (block_size, 1, 1)

        surviving_preys(actual_preys_gpu, actual_predators_gpu, collision_gpu, np.int32(radius), np.int32(predators_num),
                        block=block_size, grid=grid_size)

        collision_result = np.empty_like(collision)
        cuda.memcpy_dtoh(collision_result, collision_gpu)

        return np.where(collision_result.reshape((preys_num, predators_num)) == 1)
    return np.array([[]])


def get_vision(predators, preys):
    if predators and preys:
        find_angles = mod.get_function("find_angles")
        actual_predators = np.array(
            [(predator.position[0], predator.position[1], predator.get_view_angle()) for predator in
             predators]).flatten().astype(np.float64)
        actual_entities = np.concatenate(
            (np.array([prey.position for prey in preys]),
             np.array([predator.position for predator in predators]))).flatten().astype(np.float64)
        nbr_predators = len(predators)
        nbr_preys = len(preys)
        nbr_entities = nbr_preys + nbr_predators
        result = np.zeros(nbr_predators * nbr_entities * 2).astype(np.int32)

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

        return final_result
    return np.array([])


def make_acts():
    while continue_counting:
        lock.acquire()
        for i in range(len(predators)):
            predators[i].act()
            if predators[i].get_energy() == 0 or predators[i].is_dead():
                predators[i].dies()
                dead_predators.append(i)
        lock.release()
        for entity in preys:
            entity.act()
        time.sleep(1 / 12)


def update_preys():
    preys_index = 0
    predators_index = 1
    collision = get_collision(preys, predators)
    j = 0
    for i in range(collision[preys_index].size):
        collided_prey, collided_predator = collision[preys_index][i], collision[predators_index][i]
        if predators[collided_predator].in_range_of_view(preys[collided_prey - j].get_position()):
            predators[collided_predator].deal_damage()
            preys[collided_prey - j].receive_damage()
            if preys[collided_prey - j].is_dead():
                predators[collided_predator].eat_prey()
                last_gen_preys.append(preys[collided_prey - j])
                del preys[collided_prey - j]
                j += 1
                continue
            predators[collided_predator].stop()
        if preys[collided_prey - j].in_range_of_view(predators[collided_predator].get_position()):
            preys[collided_prey - j].deal_damage()
            predators[collided_predator].receive_damage()
            if predators[collided_predator].is_dead():
                preys[collided_prey - j].kill_predator()
                continue
            preys[collided_prey - j].stop()


def handle_vision(predators, preys):
    vision = get_vision(predators, preys)
    enemies_index = len(predators)
    for i in range(vision.shape[0]):
        predators[i].set_nearby_objects(vision[i], enemies_index)


def fill_with_randoms(actual_number, entity):
    if entity is Predator:
        needed = 100 - actual_number
        predators.extend([Predator(red, [random.randint(radius, game_width - radius),
                                         random.randint(radius, game_height - radius)], i,
                                   brain=[np.random.uniform(-1, 1, size=input_size), np.random.uniform(-1, 1, bias_size)],
                                   radius=radius, velocity=velocity)
                            for i in range(needed)])
    elif entity is Prey:
        needed = 100 - actual_number
        preys.extend([Prey(neon_green, [random.randint(radius, game_width - radius),
                                 random.randint(radius, game_height - radius)], i,
                           brain=[np.random.uniform(-1, 1, size=input_size), np.random.uniform(-1, 1, bias_size)],
                           radius=radius, velocity=velocity)
                      for i in range(needed)])


def init_all_images():
    prey_img = pygame.transform.smoothscale(pygame.image.load("assets/pictures/prey.png").convert_alpha(), (7, 4))
    predator_img = pygame.transform.smoothscale(pygame.image.load("assets/pictures/predator.png").convert_alpha(), (7, 4))

    prey_img_angles = [pygame.transform.rotate(prey_img, i).convert_alpha() for i in range(361)]
    predator_img_angles = [pygame.transform.rotate(predator_img, i).convert_alpha() for i in range(361)]

    return prey_img_angles, predator_img_angles


def update_predators_list():
    x = 0
    lock.acquire()
    for i in dead_predators:
        last_gen_predators.append(predators[i - x])
        del predators[i - x]
        x += 1
    lock.release()


def evolve_entities():
    brain, position = 0, 1
    current_generation = generation
    while not evolution_system.is_ended():
        time.sleep(14)
        next_predators_gen = []
        next_preys_gen = []
        if len(predators) < max_predators:
            new_predators = evolution_system.evolve(predators + last_gen_predators, Predator)
            next_predators_gen = [Predator(red, new_predators[position][i], np.random.randint(low=1, high=100),
                                           brain=new_predators[brain][i], radius=radius, velocity=velocity) for i in
                                  range(len(new_predators[brain]))]
        if len(preys) < max_preys:
            new_preys = evolution_system.evolve(preys + last_gen_preys, Prey)
            next_preys_gen = [Prey(neon_green, new_preys[position][i], np.random.randint(low=1, high=100),
                                   brain=new_preys[brain][i], radius=radius, velocity=velocity) for i in
                              range(len(new_preys[brain]))]

        last_gen_predators.clear()
        last_gen_preys.clear()

        predators.extend(next_predators_gen)
        preys.extend(next_preys_gen)
        current_generation += 1
        update_generation(current_generation)

        predators_nbr = len(predators)
        preys_nbr = len(preys)

        # to prevent entities extinction
        if predators_nbr < 100:
            fill_with_randoms(predators_nbr, Predator)
        if preys_nbr < 100:
            fill_with_randoms(preys_nbr, Prey)

        if len(predators) > 200 and len(preys) > 200:
            with open("data/saving", "wb") as f:
                saving = {
                    'predators_brains': [predator.get_brain() for predator in predators],
                    'preys_brains': [prey.get_brain() for prey in preys],
                    'generation': current_generation
                }
                pickle.dump(saving, f)


def update_generation(current_generation):
    # I feel bad for using global but i had to
    global generation
    generation = current_generation


def draw_graph(stats, total, curve_color):
    x, y = 0, 1
    offset = 5
    graph_width = 455
    graph_height = 150
    starting_point = graph_width
    graph_points = len(stats)
    white = (255, 255, 255)
    total = total
    graph_surface = pygame.Surface((graph_width, graph_height))
    graph_surface.fill((160, 160, 160))
    pygame.draw.line(graph_surface, white, (0, graph_height), (graph_width, graph_height), width=3)
    pygame.draw.line(graph_surface, white, (0, 0), (0, graph_height), width=1)
    for i in range(starting_point, (starting_point - graph_points), -1):
        pygame.draw.line(graph_surface, curve_color, (i, graph_height - offset),
                         (i, (graph_height - offset) - stats.get(starting_point - i) * 100 / total))
    return graph_surface


def elapsed_time(init_time, current_time):
    seconds_in_minute = 60
    elapsed_time = current_time - init_time
    minutes = int(elapsed_time // seconds_in_minute)
    seconds = int(elapsed_time - (elapsed_time // 60) * 60)
    if seconds < 10:
        return f"{minutes}:0{seconds}"
    return f"{minutes}:{seconds}"


def main():
    pygame.init()

    init_time = time.time()
    window = pygame.display.set_mode((window_width, window_height), pygame.FULLSCREEN)
    pygame.display.set_caption("Lotka-Volterra")
    screen = pygame.Surface((game_width, game_height))
    mini_map = pygame.Surface((minimap_width, minimap_height))
    prey_img_angles, predator_img_angles = init_all_images()

    horror_font = pygame.font.Font("assets/fonts/28DaysLater.ttf", 24)
    comic_font = pygame.font.Font("assets/fonts/TalkComic.ttf", 18)

    predators_acting_thread = threading.Thread(target=make_acts)
    predators_acting_thread.start()

    evolution_thread = threading.Thread(target=evolve_entities)
    evolution_thread.start()

    running = True

    while running:
        update_preys()
        handle_vision(predators, preys)
        handle_vision(preys, predators)
        window.fill(window_bg_color)
        screen.fill(screen_bg_color)
        mini_map.fill((240, 240, 240))
        mini_map.set_alpha(80)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for predator in predators:
            predator.move()
            predator_shape = predator_img_angles[predator.get_view_angle_in_degrees()]
            predator.draw_view_range(screen, predator_view_angles, predator_shape.get_rect().center)

            if predator.position[x] <= radius or predator.position[x] >= game_width - radius:
                predator.teleport_x(game_width)
            if predator.position[y] <= radius or predator.position[y] >= game_height - radius:
                predator.teleport_y(game_height)

            screen.blit(predator_shape, predator.position)
            mini_map.blit(predator_shape, np.array(predator.position) / 4)

        for prey in preys:
            prey.move()
            prey_shape = prey_img_angles[prey.get_view_angle_in_degrees()]
            prey.draw_view_range(screen, prey_view_angles, prey_shape.get_rect().center)

            if prey.position[x] <= radius or prey.position[x] >= game_width - radius:
                prey.teleport_x(game_width)
            if prey.position[y] <= radius or prey.position[y] >= game_height - radius:
                prey.teleport_y(game_height)

            screen.blit(prey_shape, prey.position)
            mini_map.blit(prey_shape, np.array(prey.position) / 4)

        update_predators_list()
        dead_predators.clear()

        predators_stats.push(len(predators))
        preys_stats.push(len(preys))

        predators_graph = draw_graph(predators_stats, max_predators + 100, red)
        preys_graph = draw_graph(preys_stats, max_preys + 100, neon_green)
        predators_graph.set_alpha(alpha)
        preys_graph.set_alpha(alpha)

        pygame.draw.rect(window, (255, 255, 255), (
        window_width / 1.5 - rect_width, 15 - rect_width, minimap_width + 2 * rect_width,
        minimap_height + 2 * rect_width), width=rect_width)

        fps = round(clock.get_fps())
        fps_text = horror_font.render(f"FPS {fps}", True, "white")

        current_time = time.time()
        time_text = comic_font.render(elapsed_time(init_time, current_time), True, "white")
        elapsed_time_text = comic_font.render("Elapsed Time", True, "white")

        number_of_predators = len(predators)
        number_of_preys = len(preys)

        preys_nbr_text = comic_font.render(f"Alive Preys", True, "white")
        preys_nbr = comic_font.render(f"{number_of_preys}", True, "white")
        predators_nbr_text = comic_font.render(f"Alive Predators", True, "white")
        predators_nbr = comic_font.render(f"{number_of_predators}", True, "white")

        generation_text = comic_font.render(f"Current Generation : {generation}", True, "white")
        total_entities = comic_font.render(f"Total Entities : {number_of_preys + number_of_predators}", True, "white")

        window.blit(screen, (window_width / 16, 15))
        window.blit(mini_map, (window_width / 1.5, 15))
        window.blit(predators_graph, (window_width / 1.8, minimap_height + 288))
        window.blit(predators_nbr_text, (window_width / 1.8, minimap_height + 288 - 30))
        window.blit(predators_nbr, (window_width / 1.295, minimap_height + 288 - 30))
        window.blit(preys_graph, (window_width / 1.8, minimap_height + predators_graph.get_height() + 338))
        window.blit(preys_nbr_text, (window_width / 1.8, minimap_height + predators_graph.get_height() + 338 - 30))
        window.blit(preys_nbr, (window_width / 1.295, minimap_height + predators_graph.get_height() + 338 - 30))
        window.blit(fps_text, (10, 15))
        window.blit(time_text, (window_width / 1.3, minimap_height + 15 + 8))
        window.blit(elapsed_time_text, (window_width / 1.5, minimap_height + 15 + 8))
        window.blit(generation_text, (window_width / 16 + screen.get_width() + 10, window_height / 3.3))
        window.blit(total_entities, (window_width / 16 + screen.get_width() + 10, window_height / 3.3 + generation_text.get_height() + 20))

        pygame.display.update()
        clock.tick(60)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    cProfile.run("main()", filename="stats/output.dat")

    with open("stats/output_time.txt", "w") as f:
        p = pstats.Stats("stats/output.dat", stream=f)
        p.sort_stats("time").print_stats()

    with open("stats/output_calls.txt", "w") as f:
        p = pstats.Stats("stats/output.dat", stream=f)
        p.sort_stats("calls").print_stats()
