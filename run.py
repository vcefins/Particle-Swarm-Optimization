from fn_utils import *
import matplotlib.pyplot as plt
from IPython import display
import time
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

# Initial Setup
from optimizer import Optimizer
from function_classes import SimpleConvex, Rastrigin, Rosenbrock
from PSO import *

# Define the Figure object and the 6 sub-figures (Axes)
fig, [[ax11, ax12], [ax21, ax22], [ax31, ax32]] = plt.subplots(3, 2)

###################################################################
# Simple Convex
simple_convex = SimpleConvex()

GD_1 = Optimizer()
GD_1.initialize_particles()
GD_1.set_benchmark_function(simple_convex)

swarm_1 = SwarmOptimizer(n_particles=10, v_coeff=0.5, p_coeff=1, g_coeff=0.07)
swarm_1.set_objective_function(simple_convex)
swarm_1.get_population()

X1, Y1, Z1 = F('simple_convex')
p11 = ax11.contourf(X1, Y1, Z1)
ax11.set_xlim(-2, 2)
ax11.set_ylim(-2, 2)
p12 = ax12.contourf(X1, Y1, Z1)
ax12.set_xlim(-2, 2)
ax12.set_ylim(-2, 2)

fig.colorbar(p11, ax=ax11)
fig.colorbar(p12, ax=ax12)

scatter_11 = ax11.scatter(0, 0, color='white')
scatter_12 = ax12.scatter(0, 0, color='white')

###################################################################
# Rastrigin
rastrigin = Rastrigin()

GD_2 = Optimizer()
GD_2.initialize_particles()
GD_2.set_benchmark_function(rastrigin)

swarm_2 = SwarmOptimizer(n_particles=10, v_coeff=0.5, p_coeff=1, g_coeff=0.07)
swarm_2.set_objective_function(rastrigin)
swarm_2.get_population()

X2, Y2, Z2 = F('rastrigin')
p21 = ax21.contourf(X2, Y2, Z2)
ax21.set_xlim(-2, 2)
ax21.set_ylim(-2, 2)
p22 = ax22.contourf(X2, Y2, Z2)
ax22.set_xlim(-2, 2)
ax22.set_ylim(-2, 2)

fig.colorbar(p21, ax=ax21)
fig.colorbar(p22, ax=ax22)


scatter_21 = ax21.scatter(0, 0, color='white')
scatter_22 = ax22.scatter(0, 0, color='white')

###################################################################
# Rosenbrock
rosenbrock = Rosenbrock()

GD_3 = Optimizer()
GD_3.initialize_particles()
GD_3.set_benchmark_function(rosenbrock)

swarm_3 = SwarmOptimizer(n_particles=10, v_coeff=0.5, p_coeff=1, g_coeff=0.07)
a, b = 0, 100
swarm_3.set_objective_function(rosenbrock)
swarm_3.get_population()

X3, Y3, Z3 = F('rosenbrock', rosen_a=a, rosen_b=b)
p31 = ax31.contourf(X3, Y3, Z3)
ax31.set_xlim(-2, 2)
ax31.set_ylim(-2, 2)
p32 = ax32.contourf(X3, Y3, Z3)
ax32.set_xlim(-2, 2)
ax32.set_ylim(-2, 2)

fig.colorbar(p31, ax=ax31)
fig.colorbar(p32, ax=ax32)

x_min = a
y_min = a**2
scatter_31 = ax31.scatter(x_min, y_min, color='white')
scatter_32 = ax32.scatter(x_min, y_min, color='white')

####################################################################
# DISPLAY
plt.ion()
plt.show()

for i in range(200):
    # Compute for Simple Convex
    swarm_1.step()
    next(GD_1.gradient_descent(lr=0.1))

    for particle in swarm_1.get_particles():
        ax11.scatter(*particle)
        
    for particle in GD_1.particle_coordinates:
        ax12.scatter(*particle)

    # Compute for Ratrigin
    swarm_2.step()
    next(GD_2.gradient_descent(lr=0.1))

    for particle in swarm_2.get_particles():
        ax21.scatter(*particle)
        
    for particle in GD_2.particle_coordinates:
        ax22.scatter(*particle)

    # Compute for Rosenbrock
    swarm_3.step()
    next(GD_3.gradient_descent(lr=0.005))

    for particle in swarm_3.get_particles():
        ax31.scatter(*particle)
        
    for particle in GD_3.particle_coordinates:
        ax32.scatter(*particle)


    fig.canvas.draw()
    time.sleep(0.01)
    fig.canvas.flush_events()
