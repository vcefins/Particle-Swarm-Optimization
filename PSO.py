import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 50
import function_classes
from fn_utils import *

#np.random.seed(0)

class Particle():
    ''' Particle class -- Defines a single particle '''
    def __init__(self, ID,  x_range=[-2, 2], y_range=[-2, 2]):
        self.ID = ID
        self.xrange = np.arange(x_range[0], x_range[1], 0.001)
        self.yrange =  np.arange(y_range[0], y_range[1], 0.001)
        self.position =np.array([np.random.choice(list(self.xrange)), np.random.choice(list(self.yrange))])# Current position of the particle -- Uniform Random initialization 
        self.velocity = np.random.rand(2,) # Current velocity of the particle -- Uniform Random initialization
        self.p_best_pos = self.position    # Previous best position (coordinates) of this particle
        self.p_best_val = None             # Previous best value -- Function value at p_best_val

    # Getter functions
    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_p_best_pos(self):
        return self.p_best_pos

    def get_p_best_val(self):
        return self.p_best_val

    # Setter functions
    def set_position(self, position):
        self.position = position

    def set_velocity(self, velocity):
        self.velocity = velocity

    def set_p_best_pos(self, p_best_pos):
        self.p_best_pos = p_best_pos


class SwarmOptimizer():
    ''' 
    Swarm class -- Defines the optimization process using multiple Particles

    Properties :
        Global neighbourhood - All particles have the knowledge of the global best position at each step
    '''
    def __init__(self, n_particles=20, v_coeff=0.2, p_coeff=1, g_coeff=0.07, x_range=[-2, 2], y_range=[-2, 2]):
        self.v_coeff = v_coeff
        self.p_coeff = p_coeff
        self.g_coeff = g_coeff

        self.xrange = np.arange(x_range[0], x_range[1], 0.001)
        self.yrange =  np.arange(y_range[0], y_range[1], 0.001)

        self.g_best_pos = np.array([np.random.choice(list(self.xrange)), np.random.choice(list(self.yrange))])    # Previous best coordinates of the (global) neighbourhood
        self.g_best_val = None      # Previous best value of the (global) neighbourhood -- Function value at g_best_pos

        self.population = [] # All particles in this swarm
        for p in range(n_particles):
            particle = Particle(p, x_range=x_range, y_range=y_range)
            self.population.append(particle)

    def set_objective_function(self, function):
        self.function = function
        # Set the initial Global best position of the swarm -- g_best_pos
        self.g_best_val = self.function.get_value(*self.g_best_pos)
        for particle in self.population:
            p_best_pos = particle.get_p_best_pos()
            if self.function.get_value(*p_best_pos) < self.g_best_val:
                self.g_best_pos = particle.position
                self.g_best_val = self.function.get_value(*self.g_best_pos)



    def get_particles(self): # Yield particles' coordinates
        X_coords, Y_coords = [], []
        for particle in self.population:
            yield particle.get_position()
    
    def get_population(self): # Return particles' coordinates
        particles = []
        for particle in self.population:
            particles.append(particle.get_position())
        return particles

    def step(self):
        for particle in self.population:
            pos = particle.get_position() # Get current position
            vel = particle.get_velocity() # Get current velocity
            p_best_pos = particle.get_p_best_pos() # Get current particle's best position
            # Update velocity
            vel = self.v_coeff * vel + \
                  self.p_coeff * np.random.rand(2,) * (p_best_pos - pos) + \
                  self.g_coeff * np.random.rand(2,) * (self.g_best_pos - pos)

            pos += vel # Update position

            particle.set_velocity(vel) # Write new velocity
            particle.set_position(pos) # Write new position

            if self.function.get_value(*pos) <= self.function.get_value(*p_best_pos):
                p_best_pos = pos
                #print("updating p_best")
                particle.set_p_best_pos(p_best_pos)
                if self.function.get_value(*p_best_pos) < self.g_best_val:
                    #print("updating g_best")
                    self.g_best_pos = p_best_pos


    def optimize(self, n_iter):
        # Start iterative optimization
        for i in range(n_iter):
            self.step()


if __name__ == '__main__':

    PSO = SwarmOptimizer(n_particles=20, v_coeff=0.2, p_coeff=1, g_coeff=0.09)
    a, b = 0, 100
    function = function_classes.Rosenbrock(a, b)
    PSO.set_objective_function(function)

    fig = plt.figure()
    X, Y, Z = F('rosenbrock')
    p = plt.pcolor(X, Y, Z)
    fig.colorbar(p)
    x_min = np.min(Z, axis=0)
    y_min = np.min(Z, axis=1)
    plt.scatter(0, 0, color='white')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)


    PSO.optimize(n_iter=300)


    particles = np.array(PSO.get_population())
    plt.scatter(particles[:,0], particles[:,1])
    plt.show()

    p = particles[0]
    print("Particle position: ", p.get_position())
    print("Global minimum at: ({},{})".format(a,a**2))
    print("Value at global minimum: ", function.get_value(a,a**2))
