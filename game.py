from utils import *
from car import Car
from pyglet.gl import gl
import tensorflow as tf
import tflearn
import numpy as np

class Game:

    def start(self):
        def update(dt,batch):
            for obj in game_objects:
                obj.update(dt,batch)

        pyglet.resource.path = ['resources']
        pyglet.resource.reindex()

        #game_window = pyglet.window.Window()
        game_window = pyglet.window.Window(width=1280,height=720)
        gl.glClearColor(.25,.25,.25,1)
        batch = pyglet.graphics.Batch()

        car_image = pyglet.resource.image("car.png")
        center_image(car_image)

        score_label = pyglet.text.Label(text="Rotations: 0", x=10, y=575, batch=batch)
        distance_label = pyglet.text.Label(text="Distance: 0", x=10, y=555, batch=batch)
        speed_label = pyglet.text.Label(text="Speed: 0", x=10, y=535, batch=batch)
        player_car = Car(img=car_image,x=200,y=200, batch=batch)
        player_car.scale = CAR_SIZE
        game_window.push_handlers(player_car)
        game_objects = [player_car]

        @game_window.event
        def on_draw():
            game_window.clear()
            batch.draw()
            score_label.text = "Rotations: {}".format(player_car.getScore())
            distance_label.text = "Distance: {}".format(player_car.getDistance())
            speed_label.text = "Speed: {}".format(player_car.getSpeed())

        pyglet.clock.schedule_interval(update,1/144.0,batch)
        pyglet.app.run()
