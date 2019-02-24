from utils import *
from car import Car
from pyglet.gl import gl
import tensorflow as tf
import tflearn
import numpy as np
import QCOPY as ql
import time

class Game:
    r = False
    score = 0
    def start(self):
        def update(dt,batch):
            distance = player_car.getDistance()
            speed = player_car.getSpeed()
            rotationspeed = player_car.getRotationSpeed()
            rotation = player_car.getRotation()
            action = RL.choose_action("d:{},s:{},rs:{},r:{}".format(str(round(distance,0)),str(round(speed,0)),str(round(rotationspeed,0)),str(round(rotation,0))))
            player_car.action(action)
            for obj in game_objects:
                obj.update(dt,batch)
            player_car.actionreset()
            reward = player_car.getScore() - self.score
            self.score = player_car.getScore()
            print("{} -- {}".format(action,reward))
            _distance = player_car.getDistance()
            _speed = player_car.getSpeed()
            _rotationspeed = player_car.getRotationSpeed()
            _rotation = player_car.getRotation()
            if(distance>100 or reward>10):
                player_car.reset(200,200)
            else:
                done = False
            RL.learn("d:{},s:{},rs:{},r:{}".format(str(round(distance,0)),str(round(speed,0)),str(round(rotationspeed,0)),str(round(rotation,0))),action,reward, "d:{},s:{},rs:{},r:{}".format(str(round(_distance,0)),str(round(_speed,0)),str(round(_rotationspeed,0)),str(round(_rotation,0))))
        
        def updateh(dt,batch):
            for obj in game_objects:
                obj.update(dt,batch)

        def updatel(dt,batch):
            if(self.r != True):
                learn(dt,batch)
                self.r = True
            else:
                print("E")

        def learn(dt,batch):
            for episode in range(100):
                player_car.reset(200,200)
                distance = player_car.getDistance()
                speed = player_car.getSpeed()
                done = False
                
                while True:
                    action = RL.choose_action(str(distance))
                    
                    player_car.action(action)
                    update(1/144,batch)
                    player_car.actionreset()
                    
                    reward = player_car.getScore() + -1*(player_car.getDistance()/100.0)*.5
                    _distance = player_car.getDistance()
                    _speed = player_car.getSpeed()
                    if(distance>1000 or reward>10):
                        done = True
                    else:
                        done = False
                    
                    print(action)
                    print(distance)
                    print(reward)

                    RL.learn(str(distance),action,reward, str(_distance))

                    distance = _distance
                    speed = _speed

                    if done:
                        break

                    



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
            try:
                fps_label.text = "fps: {}".format(1/player_car.getDT())
            except:
                pass

            

        
        RL = ql.QLearningTable(actions=['u','d','l','r','n'])
        pyglet.clock.schedule_interval(update,1/144.0,batch)
        #pyglet.clock.schedule_once(learn,1/144.0,batch)
        pyglet.app.run()
