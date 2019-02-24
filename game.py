from utils import *
from car import Car
from pyglet.gl import gl
import tensorflow as tf
import tflearn
import numpy as np
import QCOPY as ql
import time
import random

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
            with tf.Session() as sess:
                sess.run(self.init)
                for i in range(self.num_episodes):
                    player_car.reset(200,200)
                    distance = player_car.getDistance()
                    speed = player_car.getSpeed()
                    rotationspeed = player_car.getRotationSpeed()
                    rotation = player_car.getRotation()
                    s = np.array([distance, speed, rotationspeed, rotation])
                    s = s.reshape(1,4)
                    rAll = 0
                    d = False
                    j = 0

                    while j < 4000:
                        j+=1
                        
                        a,allQ = sess.run([self.predict, self.Qout], feed_dict = {self.inputs1:s})
                        if np.random.rand(1)<self.e:
                            a[0] = random.randint(0,4)# random action
                        

                        player_car.action(a[0])
                        updateh(1/144,batch)
                        player_car.actionreset()

                        _distance = player_car.getDistance()
                        _speed = player_car.getSpeed()
                        _rotationspeed = player_car.getRotationSpeed()
                        _rotation = player_car.getRotation()
                        s1 = np.array([_distance, _speed, _rotationspeed, _rotation])
                        s1 = s1.reshape(1,4)

                        

                        #r = player_car.getScore()
                        r = player_car.getScore() - self.score
                        self.score = player_car.getScore()

                        d = False
                        if(distance>100 or r > 10):
                            d = True
                        if(j%100==0):
                            print(sess.run(self.W))
                            print(a,allQ,self.predict, self.Qout,self.nextQ,s,self.W)
                            print(round(distance,2),round(r,2),round(self.score,2),i,j)

                        Q1 = sess.run(self.Qout,feed_dict={self.inputs1:s1})
                        maxQ1 = np.max(Q1)
                        targetQ = allQ
                        targetQ[0,a[0]] = r + self.y*maxQ1

                        _,W1 = sess.run([self.updateModel,self.W], feed_dict={self.inputs1:s,self.nextQ:targetQ})
                        rAll += r
                        distance = _distance
                        speed = _speed
                        rotationspeed = _rotationspeed
                        rotation = _rotation
                        s = s1
                        if d==True:
                            self.e = 1./((i/50)+10)
                            break
                    self.jList.append(j)
                    self.rList.append(rAll)
            print("Percent of succesful episodes: " + str(sum(self.rList)/self.num_episodes) + "%")


            

                    



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

            

        
        #RL = ql.QLearningTable(actions=['u','d','l','r','n'])
        tf.reset_default_graph()
        self.inputs1 = tf.placeholder(shape=[1,4], dtype = tf.float32)
        self.W = tf.Variable(tf.random_uniform([4,5],0,1))
        self.Qout = tf.matmul(self.inputs1,self.W)
        self.predict = tf.argmax(self.Qout,1)


        self.nextQ = tf.placeholder(shape=[1,5],dtype = tf.float32)
        self.nextQ = tf.Print(self.nextQ,[self.nextQ],"nextQ: ")
        loss = tf.reduce_sum(tf.square(self.nextQ-self.Qout)+1e50)
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.updateModel = trainer.minimize(loss)
        self.init = tf.global_variables_initializer()
        print(self.predict)

        self.y = .99
        self.e = .1
        self.num_episodes = 500
        self.jList = []
        self.rList = []


        #pyglet.clock.schedule_interval(update,1/144.0,batch)
        pyglet.clock.schedule_once(learn,1/144.0,batch)
        pyglet.app.run()
