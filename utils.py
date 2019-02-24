import pyglet

CAR_SIZE = 0.03
CAR_TURN_RATE = 10000000

CAR_ENGINE_FORCE = 100000
CAR_DRAG = .1
CAR_MASS = 300

def center_image(image):
    image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2