"""Test objectives for Track A. Each has f(x), grad(x) and hess(x) written out by hand."""
import numpy as np


class Quadratic:
    """Convex quadratic f(x) = 0.5 x^T A x. Ill-conditioned: condition number = 100.
    Unique global minimum at x = 0, f* = 0."""
    name = "quadratic"
    # Asymmetric start: from (1, 1) AdaGrad/RMSProp with lr=1 step by exactly sign(g) and land on 0 by accident
    x0 = np.array([1.3, 0.7])
    f_star = 0.0

    def __init__(self, cond=100.0):
        self.A = np.diag([1.0, cond])

    def f(self, x):
        return 0.5 * x @ self.A @ x

    def grad(self, x):
        return self.A @ x

    def hess(self, x):
        return self.A


class Rosenbrock:
    """Non-convex 'banana valley' f = (1-x)^2 + 100 (y - x^2)^2.
    Global minimum at (1, 1), f* = 0. Hard because the valley is long, narrow and curved."""
    name = "rosenbrock"
    x0 = np.array([-1.2, 1.0])
    f_star = 0.0

    def f(self, p):
        x, y = p
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    def grad(self, p):
        x, y = p
        return np.array([-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)])

    def hess(self, p):
        x, y = p
        return np.array([[2 - 400 * (y - x ** 2) + 800 * x ** 2, -400 * x],
                         [-400 * x, 200.0]])


class Saddle:
    """Non-convex f = x^2 + y^4/4 - y^2/2.
    Saddle point at (0, 0) (f = 0); two global minima at (0, +-1) (f* = -0.25).
    We start almost exactly on the ridge (y = 1e-6) to show how methods escape (or fail to escape) the saddle."""
    name = "saddle"
    x0 = np.array([1.0, 1e-6])
    f_star = -0.25

    def f(self, p):
        x, y = p
        return x ** 2 + y ** 4 / 4 - y ** 2 / 2

    def grad(self, p):
        x, y = p
        return np.array([2 * x, y ** 3 - y])

    def hess(self, p):
        x, y = p
        return np.array([[2.0, 0.0], [0.0, 3 * y ** 2 - 1]])


ALL = [Quadratic(), Rosenbrock(), Saddle()]
