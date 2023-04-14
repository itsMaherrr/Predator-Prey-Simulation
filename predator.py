from entity import Entity


class Predator(Entity):
    def __init__(self, color, x_pos, id, y_pos, brain, radius=2, velocity=1):
        Entity.__init__(self, color, x_pos, id, y_pos, brain, radius, velocity)
        self.eaten_preys = 0
        if self.id == 25:
            self.color = (255, 255, 255)

    def deal_damage(self):
        self.hp -= 25

    def eat_prey(self, number_of_preys):
        self.eaten_preys += number_of_preys