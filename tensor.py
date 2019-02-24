import game as g
import tensorflow as tf
import tflearn
import numpy as np

initial_games = 100
test_games = 100
goal_steps = 100
lr = 1e-2

game = g.Game()
game.start()
