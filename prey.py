from entity import Entity


class Prey(Entity):
    def __init__(self, color, x_pos, id, y_pos, brain, radius=2, velocity=1):
        Entity.__init__(self, color, x_pos, id, y_pos, brain, radius, velocity)
        if id == 25:
            self.color = (0, 0, 0)

    def deal_damage(self):
        self.hp -= 50