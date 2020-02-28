import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
# Simple bi-variate convex function with global minimum at (0,0)
def simple_convex(X, Y, *args):
    Z = X**2 + Y**2
    return Z

# Rastrigin Function
def rastrigin(X, Y, *args):
    Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + \
    (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
    return Z

# Rosenbrock Function
def rosenbrock(X, Y, a, b):
    Z = (a-X)**2 + b*(Y-X**2)**2
    return Z


# Function selector
def F(function, rosen_a=1, rosen_b=100, X=np.linspace(-2, 2, 100), Y=np.linspace(-2, 2, 100)):
	if isinstance(X, np.ndarray) and isinstance(X, np.ndarray):
		X, Y = np.meshgrid(X, Y)
	elif isinstance(X, float) and isinstance(X, float):
		pass
	else:
		print("Only Float and Numpy arrays accepted")
		exit()

	Z = getattr(sys.modules[__name__], function)(X, Y, rosen_a, rosen_b)
	return X, Y, Z
