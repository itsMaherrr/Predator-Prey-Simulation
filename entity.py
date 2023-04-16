from abc import ABC, abstractmethod

import numpy as np
import pygame
from pygame.sprite import Sprite
import random
import time
import sched
import threading
import math


class Entity(ABC, Sprite):
    x = 0
    y = 1
    one_second = 1
    view_range = 1000

    def __init__(self, color, x_pos, id, y_pos, brain, radius=2, velocity=1):
        Sprite.__init__(self)
        self.id = id
        self._nearby_objects = np.array([[0, 0]])
        self._enemies_nbr = 0
        self._brain = brain
        self._eyes = np.ones(48)
        self.color = color
        self.position = [x_pos, y_pos]
        self.radius = radius
        self.linear_speed = random.randint(-velocity, velocity)
        self.angular_velocity = - math.pi / 180
        self._view_angle = np.random.randint(360)
        self.draw_lines = False
        self.hp = 100
        self._survival_time = 0
        self._scheduler = sched.scheduler(time.time, time.sleep)
        #self.counter = threading.Thread(target=self.act)
        #self.counter.start()

    def move(self):
        #print(f"linear speed is {self.linear_speed}, view angle is {self._view_angle} and cos is {math.cos(self._view_angle)}")
        self.position[Entity.x] += self.linear_speed * math.cos(self._view_angle)
        self.position[Entity.y] += self.linear_speed * math.sin(self._view_angle)
        self.update_view_angle()

    def rebound(self):
        self.linear_speed = -self.linear_speed

    def __reverse_x_velocity(self):
        self.linear_speed = -self.linear_speed

    """
    def __reverse_y_velocity(self):
        self.velocity[Entity.y] = -self.velocity[Entity.y]
    """

    def change_velocity(self, linear_speed, angular_velocity):
        self.linear_speed = linear_speed
        self.angular_velocity = angular_velocity

    def update_eyes(self, inputs):
        self._eyes = np.array(inputs)

    def think_and_move(self):
        potential = np.dot(self._brain, self._eyes)
        linear_speed, angular_velocity = np.tanh(potential[0]), np.tanh(potential[1])
        self.change_velocity(linear_speed * 2, angular_velocity / 10)

    def get_survival_time(self):
        return self._survival_time

    def set_survival_time(self, survival_time):
        self._survival_time = survival_time

    survival_time = property(get_survival_time, set_survival_time)

    def increment_survival_time(self):
        self.survival_time += 1/30

    def act(self):
        #while self._continue_counting:
            #time.sleep(0.5)
            self.increment_survival_time()
            self.handle_nearby_entities()
            self.think_and_move()
            self.update_view_angle()

    def update_view_angle(self):
        self._view_angle = (self._view_angle + self.angular_velocity) % (-2 * math.pi)

    def stop_counting(self):
        self._continue_counting = False

    def set_nearby_objects(self, objects):
        self._nearby_objects = objects

    def get_points_at_angles_and_distance(self, distance, angles):
        points = []
        for angle in angles:
            radians = math.radians(angle)
            x = self.position[self.x] + distance * math.cos(radians)
            y = self.position[self.y] + distance * math.sin(radians)
            points.append((x, y))
        return points

    def get_view_angle_in_degrees(self):
        return - int(math.degrees(self._view_angle))

    def get_view_angle(self):
        return self._view_angle

    def draw_lines_on_off(self):
        self.draw_lines = not self.draw_lines

    def draw_view_range(self, screen, angles, start):
        if self.draw_lines:
            white = (255, 255, 255)
            base_angle = - self.get_view_angle_in_degrees()
            relative_angles = [base_angle + angle for angle in angles]
            points = self.get_points_at_angles_and_distance(self.view_range, relative_angles)
            if self.id == 0:
                for point in points:
                    start_position = (self.position[Entity.x] + start[Entity.x], self.position[Entity.y] + start[Entity.y])
                    pygame.draw.aaline(screen, white, start_position, point, 1)

    @abstractmethod
    def handle_nearby_entities(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    @abstractmethod
    def deal_damage(self):
        pass