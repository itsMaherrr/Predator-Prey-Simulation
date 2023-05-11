from entity import Entity
import numpy as np

view_angles = np.linspace(-23, 23, num=24) % 360
inputs_map = {view_angles[i]: 2 * int(i) for i in range(Entity.inputs_nbr // 2)}


class Predator(Entity):

    def __init__(self, color, position, id, brain, radius=2, velocity=1):
        Entity.__init__(self, color, position, id, brain, radius, velocity)
        self.eaten_preys = 0
        self.__energy = 100
        self.__received_damage = 0
        self.damage_dealt = 0

    def deal_damage(self):
        self.damage_dealt += 25

    def receive_damage(self):
        self.hp -= 12

    def get_received_damage(self):
        return self.__received_damage

    def act(self):
        super().act()
        self.__energy = max(0, self.__energy - 0.5)

    def get_energy(self):
        return self.__energy

    def get_eaten_preys(self):
        return self.eaten_preys

    def eat_prey(self):
        self.eaten_preys += 1
        self.hp = min(100, self.hp + 10)
        self.__energy = min(100, self.__energy + 60)

    def get_damage_dealt(self):
        return self.damage_dealt

    def think_and_move(self):
        brain = self._brain[0]
        bias = self._brain[1]
        potential = np.dot(brain, self._eyes)
        linear_speed, angular_velocity = self.sigmoid(potential[0] + bias[0]), np.tanh(potential[1] + bias[1])
        self.change_velocity(linear_speed, angular_velocity / 12)

    def in_range_of_view(self, entity_position):
        x, y = 0, 0
        view_angle = self.get_view_angle()
        self_x = np.cos(view_angle)
        self_y = np.sin(view_angle)
        vect_x = entity_position[x] - self.position[x]
        vect_y = entity_position[y] - self.position[y]
        angle_between = np.degrees(np.arctan2(self_x * vect_y - self_y * vect_x, self_x * vect_x + self_y * vect_y))
        return not 45 < angle_between % 360 < 315

    def handle_nearby_entities(self):
        entities = np.array(self._nearby_objects)
        in_view_range_idx = np.where(np.isin(entities[:, 0], view_angles))
        inputs = [1 / self.view_range if i % 2 == 0 else 0 for i in range(Predator.inputs_nbr)]
        for i in range(in_view_range_idx[0].size):
            index = in_view_range_idx[0][i]
            distance = entities[index, 1]
            if distance != 0:
                vision_angle = entities[index, 0]
                input_index = inputs_map.get(vision_angle)
                distance = 1 / distance
                if inputs[input_index] <= distance:
                    type = 1 if index >= self._enemies_index else -1
                    inputs[input_index] = distance
                    inputs[input_index + 1] = type
        self.update_eyes(np.array(inputs))
