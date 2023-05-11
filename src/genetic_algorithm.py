import time
import numpy as np
from predator import Predator


class Evolution:
    max_generations = 10000

    def __init__(self):
        self.__generation_number = 0

    def is_ended(self):
        return self.__generation_number > Evolution.max_generations

    @staticmethod
    def __measure_predators_fitness(population):
        fitness = []
        t = time.time()
        for individual in population:
            survival_time = individual.get_survival_time(t) // 14
            preys_eaten = individual.get_eaten_preys()
            # damage_dealt = individual.get_damage_dealt() / 100
            individual_fitness = survival_time * 1 + preys_eaten * 3
            fitness.append((individual, individual_fitness))
        return fitness

    @staticmethod
    def __measure_preys_fitness(population):
        fitness = []
        t = time.time()
        for individual in population:
            survival_time = individual.get_survival_time(t)
            predators_killed = individual.get_killed_predators()
            individual_fitness = survival_time * 3 + 1 * predators_killed
            fitness.append((individual, individual_fitness))
        return fitness

    @staticmethod
    def __selection(measured_fitness, population_type):
        population = len(measured_fitness)
        if population_type == Predator:
            low = min(max(6, population // 75), population - 1)
            high = min(max(8, population // 65), population - 1)
        else:
            low = min(max(7, population // 65), population - 1)
            high = min(max(9, population // 55), population - 1)
        high = high + int(low >= high)
        density = np.random.randint(low=low, high=high)
        return sorted(measured_fitness, key=lambda x: x[1], reverse=True)[:density]

    @staticmethod
    def __crossover(brain1, brain2):
        brain = 0
        bias = 1

        inputs_size = 48

        crossover_point = np.random.randint(1, inputs_size - 1)
        bias_crossover_point = 1

        child1_brain = [
            np.concatenate([brain1[brain][:, :crossover_point], brain2[brain][:, crossover_point:]], axis=1),
            np.concatenate([brain1[bias][:bias_crossover_point], brain2[bias][bias_crossover_point:]])]
        child2_brain = [
            np.concatenate([brain2[brain][:, :crossover_point], brain1[brain][:, crossover_point:]], axis=1),
            np.concatenate([brain2[bias][:bias_crossover_point], brain1[bias][bias_crossover_point:]], axis=0)]

        return child1_brain, child2_brain

    @staticmethod
    def __mutation(brain):
        brain_length = 48
        column = np.random.randint(low=0, high=brain_length - 1)
        row = np.random.randint(low=0, high=1)
        where = brain_length * row + column
        np.put(brain[0], where, np.random.uniform(-1, 1))
        return brain

    @staticmethod
    def __crossover_position(position1, position2):
        x, y = 0, 1
        return [position1[x], position2[y]], [position2[x], position1[y]]

    def evolve(self, population, population_type):
        mutation_prob = 0.05
        crossover_prob = 0.8
        entity, fitness = 0, 1
        if population_type == Predator:
            fitness = self.__measure_predators_fitness(population)
            selected = self.__selection(fitness, population_type)
        else:
            fitness = self.__measure_preys_fitness(population)
            selected = self.__selection(fitness, population_type)

        evolved_brains = []
        evolved_positions = []
        for individual1 in selected:
            for individual2 in selected:
                if np.random.random() < crossover_prob:
                    brain1, brain2 = self.__crossover(individual1[entity].get_brain(), individual2[entity].get_brain())
                    position1, position2 = self.__crossover_position(individual1[entity].get_position(),
                                                                     individual2[entity].get_position())
                    if np.random.random() < mutation_prob:
                        brain1 = self.__mutation(brain1)
                    if np.random.random() < mutation_prob:
                        brain2 = self.__mutation(brain2)

                    evolved_brains.append(brain1)
                    evolved_positions.append(position1)
                    evolved_brains.append(brain2)
                    evolved_positions.append(position2)

        evolved_positions = np.array(evolved_positions)

        return evolved_brains, evolved_positions
