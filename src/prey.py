from entity import Entity
import numpy as np

view_angles = np.linspace(-105, 102, num=24) % 360
inputs_map = {view_angles[i]: 2 * int(i) for i in range(Entity.inputs_nbr // 2)}


class Prey(Entity):

    def __init__(self, color, position, id, brain, radius=2, velocity=1):
        Entity.__init__(self, color, position, id, brain, radius, velocity)
        if id == 25:
            self.color = (0, 0, 0)
        self.damage_dealt = 0
        self.predators_killed = 0

    def deal_damage(self):
        self.damage_dealt += 12


    def receive_damage(self):
        self.hp -= 25

    def get_killed_predators(self):
        return self.predators_killed

    def kill_predator(self):
        self.predators_killed += 1

    def get_damage_dealt(self):
        return self.damage_dealt

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
        inputs = [1 / self.view_range if i % 2 == 0 else 0 for i in range(Prey.inputs_nbr)]
        for i in range(in_view_range_idx[0].size):
            index = in_view_range_idx[0][i]
            distance = entities[index, 1]
            if distance != 0:
                vision_angle = entities[index, 0]
                input_index = inputs_map.get(vision_angle)
                distance = 1 / distance
                if inputs[input_index] <= distance:
                    type = -1 if index >= self._enemies_index else 1
                    inputs[input_index] = distance
                    inputs[input_index + 1] = type
        self.update_eyes(np.array(inputs))
