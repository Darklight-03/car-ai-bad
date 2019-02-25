from utils import *
from car import Car
from pyglet.gl import gl
import tensorflow as tf
import tflearn
import numpy as np
import DQNCOPY as ql
import time
import matplotlib.pyplot as plt


class Game:
    r = False
    score = 0
    done =  True
    step = 0
    attempt = 0
    scores = []
    y = []
    def start(self):

        
        def updateh(dt,batch):
            for obj in game_objects:
                obj.update(dt,batch)
            self.r = (player_car.getScore() + -1*(player_car.getDistance()/100.0)*.5)

                    



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
        rotationspeed_label = pyglet.text.Label(text="Rotation Speed: 0", x=10, y=515, batch=batch)
        direction_label = pyglet.text.Label(text="Direction: 0", x=10, y=495, batch=batch)
        fps_label = pyglet.text.Label(text="fps: 0", x=10, y=475, batch=batch)
        attempts_label = pyglet.text.Label(text="Attempt #: 0", x=10, y=615,batch=batch)
        sc_label = pyglet.text.Label(text="Score: 0", x=10,y=435,batch=batch)

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
            direction_label.text = "Direction: {}".format(player_car.getRotation())
            rotationspeed_label.text = "Rotation Speed: {}".format(player_car.getRotationSpeed())
            attempts_label.text = "Attempt #: {}".format(self.attempt)
            sc_label.text = "Score: {}".format(round(self.r,2))
            try:
                fps_label.text = "fps: {}".format(1/player_car.getDT())
            except:
                pass

            

        
        RL = ql.DeepQNetwork(5,4,learning_rate=0.0001,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2,
                      # output_graph=True
                      )

        pyglet.clock.schedule_interval(updateh,1/144.0,batch)
        pyglet.app.run()
