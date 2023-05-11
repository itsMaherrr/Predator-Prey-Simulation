from abc import ABC, abstractmethod

import numpy as np
import pygame
import random
import time
import math


class Entity(ABC):
    x = 0
    y = 1
    one_second = 1
    view_range = 2000
    inputs_nbr = 48

    def __init__(self, color, position, id, brain, radius=2, velocity=1):
        self.id = id
        self._nearby_objects = np.array([[0, 0]])
        self._enemies_index = 0
        self._brain = brain
        self._eyes = np.ones(48)
        self.color = color
        self.position = position
        self.radius = radius
        self.linear_speed = random.randint(-velocity, velocity)
        self.angular_velocity = - math.pi / 180
        self._view_angle = np.random.randint(360)
        self.draw_lines = False
        self.death_time = np.inf
        self.hp = 100
        self._brith_time = time.time()

    def move(self):
        self.position[Entity.x] += self.linear_speed * math.cos(self._view_angle)
        self.position[Entity.y] += self.linear_speed * math.sin(self._view_angle)
        self.update_view_angle()

    def dies(self):
        self.death_time = time.time()

    def get_death_time(self):
        return self.death_time

    def rebound(self, game_width, game_height):
       self.position[Entity.x] = min(game_width - self.radius, max(0, self.position[Entity.x]))
       self.position[Entity.y] = min(game_height - self.radius, max(0, self.position[Entity.y]))

    def teleport_x(self, game_width):
        self.position[Entity.x] = self.position[Entity.x] % game_width

    def teleport_y(self, game_height):
        self.position[Entity.y] = self.position[Entity.y] % game_height

    def __reverse_x_velocity(self):
        self.linear_speed = -self.linear_speed

    def get_position(self):
        return self.position

    def change_velocity(self, linear_speed, angular_velocity):
        self.linear_speed = linear_speed
        self.angular_velocity = angular_velocity

    def update_eyes(self, inputs):
        self._eyes = np.array(inputs)

    def think_and_move(self):
        brain = self._brain[0]
        bias = self._brain[1]
        potential = np.dot(brain, self._eyes)
        linear_speed, angular_velocity = np.tanh(potential[0] + bias[0]), np.tanh(potential[1] + bias[1])
        self.change_velocity(linear_speed, angular_velocity / 12)

    def get_survival_time(self, t):
        return min(t, self.get_death_time()) - self._brith_time

    def get_brith_time(self):
        return self._brith_time

    def get_brain(self):
        return self._brain

    def act(self):
        #while self._continue_counting:
            #time.sleep(0.5)
            self.handle_nearby_entities()
            self.think_and_move()
            self.update_view_angle()

    def is_dead(self):
        return self.hp <= 0

    def update_view_angle(self):
        self._view_angle = (self._view_angle + self.angular_velocity) % (-2 * math.pi)

    def set_nearby_objects(self, objects, enemies_nbr):
        self._nearby_objects = objects
        self._enemies_index = enemies_nbr

    def stop(self):
        self.linear_speed = 0

    def get_points_at_angles_and_distance(self, eyes, angles):
        points = []
        colors = []
        white = (255, 255, 255)
        red = (255, 0, 0)
        for i in range(len(angles)):
            radians = math.radians(angles[i])
            distance = 1 / eyes[2*i]
            x = self.position[self.x] + distance * math.cos(radians)
            y = self.position[self.y] + distance * math.sin(radians)
            if 0 < distance < self.view_range:
                color = red
            else:
                color = white
            points.append((x, y))
            colors.append(color)
        return points, colors

    def get_view_angle_in_degrees(self):
        return - int(math.degrees(self._view_angle))

    def get_view_angle(self):
        return self._view_angle

    def draw_lines_on_off(self):
        self.draw_lines = not self.draw_lines

    def draw_view_range(self, screen, angles, start):
        if self.draw_lines:
            base_angle = - self.get_view_angle_in_degrees()
            relative_angles = [base_angle + angle for angle in angles]
            points, colors = self.get_points_at_angles_and_distance(self._eyes, relative_angles)
            if self.id == 0:
                for i in range(len(points)):
                    start_position = (self.position[Entity.x] + start[Entity.x], self.position[Entity.y] + start[Entity.y])
                    pygame.draw.aaline(screen, colors[i], start_position, points[i], 0)

    @abstractmethod
    def handle_nearby_entities(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    @abstractmethod
    def deal_damage(self):
        pass

    @abstractmethod
    def in_range_of_view(self, entity_position):
        pass
