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
        def update(dt,batch):
            if(dt>1/10.0):
                dt = 1/10.0

            if(self.done):
                self.done = False
                player_car.reset(200,200)
                distance = player_car.getDistance()
                speed = player_car.getSpeed()
                rotationspeed = player_car.getRotationSpeed()
                rotation = player_car.getRotation()
                self.ob = np.array([distance,speed,rotationspeed,rotation])
                self.ob = self.ob.reshape(4)
            action = RL.choose_action(self.ob)
            player_car.action(action)

            for obj in game_objects:
                obj.update(dt,batch)
            player_car.actionreset()
            
            reward = (player_car.getScore() + -1*(player_car.getDistance()/100.0)*.5)# - self.score
            self.score = (player_car.getScore() + -1*(player_car.getDistance()/100.0)*.5)
            self.r = (player_car.getScore() + -1*(player_car.getDistance()/100.0)*.5)

            _distance = player_car.getDistance()
            _speed = player_car.getSpeed()
            _rotationspeed = player_car.getRotationSpeed()
            _rotation = player_car.getRotation()
            _ob = np.array([_distance,_speed,_rotationspeed,_rotation])
            _ob = _ob.reshape(4)
            if(self.ob[0]>100):
                self.done = True
                self.attempt += 1
                reward-=1
            RL.store_transition(self.ob,action,reward,_ob)
            if (self.step > 200) and (self.step % 5 == 0):
                RL.learn()
            #if(self.step%100==0):
            #    print(action,ob)
            #    print(player_car.getScore(),reward,episode)



            self.ob = _ob

            if self.done:
                
                print("done")
                print(player_car.getScore())
            self.step+=1
        
        def updateh(dt,batch):
            for obj in game_objects:
                obj.update(1/30.0,batch)
            self.r = (player_car.getScore() + -1*(player_car.getDistance()/100.0)*.5)

        #def updatec(dt,batch):


        def updatel(dt,batch):
            if(self.r != True):
                learn(dt,batch)
                self.r = True
            else:
                print("E")

        def learn(dt,batch):
            self.step = 0
            #episode = 0
            for episode in range(25000):
            #while player_car.getScore() < 8:
                #episode += 1
                self.done = False
                player_car.reset(200,200)
                distance = player_car.getDistance()
                speed = player_car.getSpeed()
                rotationspeed = player_car.getRotationSpeed()
                rotation = player_car.getRotation()
                self.ob = np.array([distance,speed,rotationspeed,rotation])
                self.ob = self.ob.reshape(4)
                
                while True:
                    action = RL.choose_action(self.ob)
                    
                    player_car.action(action)
                    updateh(1/30,batch)
                    player_car.actionreset()
                    
                    reward = ((player_car.getScore() + -1*(player_car.getDistance()/100.0)*.5) - self.score)*100
                    #print(reward)
                    self.score = (player_car.getScore() + -1*(player_car.getDistance()/100.0)*.5)
                    self.r = (player_car.getScore() + -1*(player_car.getDistance()/100.0)*.5)
                    _distance = player_car.getDistance()
                    _speed = player_car.getSpeed()
                    _rotationspeed = player_car.getRotationSpeed()
                    _rotation = player_car.getRotation()
                    _ob = np.array([_distance,_speed,_rotationspeed,_rotation])
                    _ob = _ob.reshape(4)
                    if(self.ob[0]>100 or player_car.getScore()<0 ): #
                        self.done = True
                        self.attempt += 1
                        self.score = 0
                    ee = self.score
                    while(ee>0.3):
                        reward+=0.5
                        ee-=0.3
                    eee = self.ob[0]
                    while(eee>70):
                        reward-=1
                        eee-=20
                    #print(reward)
                    RL.store_transition(self.ob,action,reward,_ob)
                    if (self.step > 200) and (self.step % 5 == 0):
                        RL.learn()
                    #if(self.step%100==0):
                    #    print(action,self.ob)
                    #    print(player_car.getScore(),reward,episode)



                    self.ob = _ob

                    if self.done:
                        print(player_car.getScore(),episode)
                        self.scores.append(player_car.getScore())
                        self.y.append(episode)
                        break
                    self.step+=1
            pyglet.clock.schedule_interval(update,1/30.0,batch)
            plt.plot (self.y,self.scores)
            plt.ylabel('end point')
            plt.xlabel('attempt #')
            plt.show()
            plt.clf()
            RL.plot_cost()

                    



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
                      memory_size=20000,
                      # output_graph=True
                      )
        #pyglet.clock.schedule_interval(update,1/144.0,batch)
        pyglet.clock.schedule_once(learn,1/144.0,batch)
        #pyglet.clock.schedule_interval(updateh,1/30.0,batch)
        pyglet.app.run()
