
import numpy as np

# Function Classes with gradients
class SimpleConvex:
    def __init__(self):
        pass

    def get_value(self, X, Y):
        Z = X**2 + Y**2
        return Z

    def grad_x(self, x, y):
        return 2 * x

    def grad_y(self, x, y):
        return 2 * y

class Rastrigin:
    def __init__(self):
        pass

    def get_value(self, X, Y):
        Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + \
        (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
        return Z

    def grad_x(self, x, y):
        return 2 * x + 440/7 * np.sin((44 * x)/7)

    def grad_y(self, x, y):
        return 2 * y + 440/7 * np.sin((44 * y)/7)

class Rosenbrock:
    def __init__(self, a=0, b=100):
        self.a = a
        self.b = b

    def get_value(self, X, Y):
        Z = (self.a-X)**2 + self.b*(Y-X**2)**2
        return Z

    def grad_x(self, x, y):
        return 2 * (self.a - x) + self.b * 2 * ( y - x**2)* x

    def grad_y(self, x, y):
        return self.b * 2 * ( y - x**2)