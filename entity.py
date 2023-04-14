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
    view_range = 200

    def __init__(self, color, x_pos, id, y_pos, brain, radius=2, velocity=1):
        Sprite.__init__(self)
        self.id = id
        self._brain = brain
        self._eyes = np.ones(10)
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
        self._continue_counting = True
        self.counter = threading.Thread(target=self.act)
        self.counter.start()

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

    def update_eyes(self):
        self._eyes = np.random.randn(10)/3

    def think_and_move(self):
        self.update_eyes()
        potential = np.dot(self._brain, self._eyes)
        linear_speed, angular_velocity = self.sigmoid(potential[0]), np.tanh(potential[1])
        self.change_velocity(linear_speed * 2, angular_velocity / 10)

    def get_survival_time(self):
        return self._survival_time

    def set_survival_time(self, survival_time):
        self._survival_time = survival_time

    survival_time = property(get_survival_time, set_survival_time)

    def increment_survival_time(self):
        self.survival_time += 1

    def act(self):
        while self._continue_counting:
            self.increment_survival_time()
            self.think_and_move()
            self.update_view_angle()
            time.sleep(1)

    def update_view_angle(self):
        self._view_angle = (self._view_angle + self.angular_velocity) % (-2 * math.pi)

    def stop_counting(self):
        self._continue_counting = False

    def detect_nearby_objects(self, objects):
        nearby_objects = []
        for obj in objects:
            distance = math.sqrt((obj.position[obj.x] - self.position[self.x]) ** 2 + (obj.position[obj.y] - self.position[self.y]) ** 2)
            if distance <= self.view_range:
                nearby_objects.append(obj)
        return nearby_objects

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

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    @abstractmethod
    def deal_damage(self):
        pass