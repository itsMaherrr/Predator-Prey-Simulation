from entity import Entity
import numpy as np


class Predator(Entity):
    view_angles = np.linspace(-23, 23, num=24) % 360

    def __init__(self, color, x_pos, id, y_pos, brain, radius=2, velocity=1):
        Entity.__init__(self, color, x_pos, id, y_pos, brain, radius, velocity)
        self.eaten_preys = 0
        if self.id == 25:
            self.color = (255, 255, 255)

    def deal_damage(self):
        self.hp -= 25

    def eat_prey(self, number_of_preys):
        self.eaten_preys += number_of_preys

    def handle_nearby_entities(self):
        inputs_nbr = 48
        inputs_map = {Predator.view_angles[i]: 2 * int(i) for i in range(inputs_nbr // 2)}
        entities = np.array(self._nearby_objects[self._nearby_objects[:, 0].argsort()]).astype(int)
        in_view_range_idx = np.where(np.isin(entities[:, 0] % 360, Predator.view_angles))
        inputs = [1 / Entity.view_range for _ in range(inputs_nbr)]
        for i in range(len(in_view_range_idx[0])):
            index = in_view_range_idx[0][i]
            vision_angle = int(entities[index, 0]) % 360
            input_index = inputs_map.get(vision_angle)
            distance = entities[index, 1]
            if distance != 0:
                distance = 1 / distance
                if inputs[input_index] < distance:
                    type = int(index < self._enemies_nbr)
                    inputs[input_index] = distance
                    inputs[input_index + 1] = type
                    if distance < 100 and 1 < input_index < inputs_nbr - 2:
                        inputs[input_index - 2] = distance
                        inputs[input_index - 1] = type
                        inputs[input_index + 2] = distance
                        inputs[input_index + 3] = type
        self.update_eyes(np.array(inputs))
