import numpy as np
import random
import logging
logging.basicConfig(level=logging.INFO)
from function_classes import *

class Optimizer:
    def __init__(self, x_range=[-2, 2], y_range=[-2, 2]):
        self.max = x_range[1]
        self.min = x_range[0]
        self.xrange = np.arange(x_range[0], x_range[1], 0.001)
        self.yrange =  np.arange(y_range[0], y_range[1], 0.001)

    def initialize_particles(self, n_particles=20):
        self.particle_coordinates = [[random.choice(list(self.xrange)), random.choice(list(self.yrange))] for idx in range(n_particles)]
        logging.info("Particles initialized at: {}".format(self.particle_coordinates))

    def gradient_descent(self, lr=0.01, iterations=10000):
        self.lr = lr
        for iter in range(iterations):
            self.update_rule()
            yield self.particle_coordinates

        
    def set_benchmark_function(self, function):
        self.benchmark_function = function

    def update_rule(self):
        for particle in self.particle_coordinates:
            particle[0] -= self.lr * self.benchmark_function.grad_x(particle[0], particle[1])
            particle[1] -= self.lr * self.benchmark_function.grad_y(particle[0], particle[1])

            if particle[0] >= self.max:
                particle[0] =  self.max

            if particle[0] <= self.min:
                particle[0] = self.min

            if particle[1] >= self.max:
                particle[1] =  self.max

            if particle[1] <= self.min:
                particle[1] = self.min

            
if __name__ == "__main__":
    opt = Optimizer()
    opt.initialize_particles()
    simple_convex = SimpleConvex()
    opt.set_benchmark_function(simple_convex)
    for idx in opt.gradient_descent():
        print(idx)