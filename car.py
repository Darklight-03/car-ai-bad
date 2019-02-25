from utils import *
import math
from physics import PhysicalObject
from pyglet.window import key

class Car(PhysicalObject):
    keys = dict(left=False,right=False,up=False,down=False)
    rotations = 0
    score = 0
    dt = 0
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        u = [math.cos(self.rotation), math.sin(self.rotation)]
        self.lines = kwargs['batch'].add(2, pyglet.gl.GL_LINES, None, ('v2d',(self.x,self.y,self.x+(u[0]*500),self.y+(u[1]*500))))

    def action(self,action):
        if(action == 1):
            self.keys['up']=True
        elif(action == 2):
            self.keys['left']=True
        elif(action == 3):
            self.keys['right']=True
        elif(action == 4):
            self.keys['down']=True

    def actionreset(self):
        self.keys['up']=False
        self.keys['left']=False
        self.keys['right']=False
        self.keys['down']=False

    def on_key_press(self,symbol,modifiers):
        if(symbol == key.UP):
            self.keys['up']=True
        elif(symbol == key.LEFT):
            self.keys['left']=True
        elif(symbol == key.RIGHT):
            self.keys['right']=True
        elif(symbol == key.DOWN):
            self.keys['down']=True
        elif(symbol == key.SPACE):
            self.reset(200,200)

    def on_key_release(self,symbol,modifiers):
        if(symbol == key.UP):
            self.keys['up']=False
        elif(symbol == key.LEFT):
            self.keys['left']=False
        elif(symbol == key.RIGHT):
            self.keys['right']=False
        elif(symbol == key.DOWN):
            self.keys['down']=False
    
    def update(self,dt,batch):
        super(Car,self).update(dt)
        engineforce = 0
        rotationforce = 0
        if(self.keys['left']):
            rotationforce -= CAR_TURN_RATE*dt
        if(self.keys['right']):
            rotationforce += CAR_TURN_RATE*dt
        if(self.keys['up']):
            engineforce += CAR_ENGINE_FORCE
        if(self.keys['down']):
            engineforce += CAR_ENGINE_FORCE*-1
        u = [math.cos(-1*math.radians(self.rotation)), -1*math.sin(math.radians(self.rotation))] # unit vector
        thrust = [u[0]*engineforce, u[1]*engineforce] # F traction -> forward
        v = [self.velocity_x, self.velocity_y] # velocity vector
        speed = math.sqrt(v[0]**2 + v[1]**2) # current speed
        drag = [-1*CAR_DRAG*v[0]*speed, -1*CAR_DRAG*v[1]*speed] # F drag -> inverse of velocity direction
        force = [thrust[0] + drag[0], thrust[1]+drag[1]]
        a = [force[0]/CAR_MASS, force[1]/CAR_MASS]
        rotationdrag = self.velocity_r*abs(self.velocity_r)*CAR_DRAG*-1
        ra = (rotationdrag+rotationforce)/CAR_MASS

        self.lines.resize(4)
        self.lines.vertices = [self.x,self.y,self.x+(u[0]*500),self.y+(u[1]*500),self.x,self.y,1280/2,720/2]
        self.velocity_x = self.velocity_x + dt * a[0]
        self.velocity_y = self.velocity_y + dt * a[1]
        self.velocity_r = self.velocity_r + dt * ra
        
        if(self.score-self.getScore() <-.97):
            self.rotations -= 1
        if(self.score-self.getScore() >.97):
            self.rotations += 1
        self.score = self.getScore()
        self.dt = dt


    def getScore(self):
        return self.rotations + (((math.atan2((self.y-(720/2)),(self.x-(1280/2)))*(180/math.pi))+180)) / 360

    def getDistance(self):
        return math.sqrt((self.y-(720/2))**2 + (self.x-(1280/2))**2)/10

    def getSpeed(self):
        return math.sqrt((self.velocity_x)**2 + (self.velocity_y)**2)/10

    def getRotationSpeed(self):
        return self.velocity_r/10
    
    def getDT(self):
        return self.dt

    def getRotation(self):
        return ((self.rotation) + (((math.atan2((self.y-(720/2)),(self.x-(1280/2)))*(180/math.pi))+180))) % 360

    def reset(self,_y,_x):
        self.x = _x
        self.y = _y
        self.velocity_x = 0
        self.velocity_y = 0
        self.velocity_r = 0
        self.rotation = 0
        self.rotations = 0