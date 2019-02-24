from utils import *

class PhysicalObject(pyglet.sprite.Sprite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.velocity_x, self.velocity_y, self.velocity_r = 0, 0, 0

    def update(self, dt):
        self.x += self.velocity_x*dt
        self.y += self.velocity_y*dt
        self.rotation += self.velocity_r*dt