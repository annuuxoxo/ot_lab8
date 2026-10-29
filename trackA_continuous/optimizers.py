"""First- and second-order optimizers written from scratch in NumPy.

Every optimizer has the same interface:
    opt.reset(dim)              clear internal state
    x_new = opt.step(x, prob)   one update; prob gives prob.grad(x) and prob.hess(x)
so they can all be run by the same loop under identical conditions.
"""
import time

import numpy as np

EPS = 1e-8


class GD:
    """Gradient descent: x <- x - lr * g."""
    name = "GD"

    def __init__(self, lr):
        self.lr = lr

    def reset(self, dim):
        pass

    def step(self, x, prob):
        return x - self.lr * prob.grad(x)


class Momentum:
    """Heavy-ball momentum: v <- beta v + g ;  x <- x - lr v.
    v is a running sum of past gradients, so steps build up speed along consistent directions."""
    name = "Momentum"

    def __init__(self, lr, beta=0.9):
        self.lr, self.beta = lr, beta

    def reset(self, dim):
        self.v = np.zeros(dim)

    def step(self, x, prob):
        self.v = self.beta * self.v + prob.grad(x)
        return x - self.lr * self.v


class NAG:
    """Nesterov accelerated gradient: like momentum, but the gradient is taken at the
    look-ahead point x - lr*beta*v, which corrects the step before overshooting."""
    name = "NAG"

    def __init__(self, lr, beta=0.9):
        self.lr, self.beta = lr, beta

    def reset(self, dim):
        self.v = np.zeros(dim)

    def step(self, x, prob):
        g = prob.grad(x - self.lr * self.beta * self.v)
        self.v = self.beta * self.v + g
        return x - self.lr * self.v


class AdaGrad:
    """AdaGrad: per-coordinate step lr / sqrt(sum of all past g^2).
    Coordinates with big gradients get smaller steps; steps only ever shrink."""
    name = "AdaGrad"

    def __init__(self, lr):
        self.lr = lr

    def reset(self, dim):
        self.G = np.zeros(dim)

    def step(self, x, prob):
        g = prob.grad(x)
        self.G += g ** 2
        return x - self.lr * g / (np.sqrt(self.G) + EPS)


class RMSProp:
    """RMSProp: like AdaGrad but uses an exponential moving average of g^2,
    so old gradients are forgotten and the step size does not decay to zero."""
    name = "RMSProp"

    def __init__(self, lr, rho=0.9):
        self.lr, self.rho = lr, rho

    def reset(self, dim):
        self.s = np.zeros(dim)

    def step(self, x, prob):
        g = prob.grad(x)
        self.s = self.rho * self.s + (1 - self.rho) * g ** 2
        return x - self.lr * g / (np.sqrt(self.s) + EPS)


class Adam:
    """Adam: momentum on g (m) + RMSProp scaling on g^2 (v), with bias correction
    because m and v start at zero."""
    name = "Adam"

    def __init__(self, lr, b1=0.9, b2=0.999):
        self.lr, self.b1, self.b2 = lr, b1, b2

    def reset(self, dim):
        self.m = np.zeros(dim)
        self.v = np.zeros(dim)
        self.t = 0

    def step(self, x, prob):
        g = prob.grad(x)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return x - self.lr * m_hat / (np.sqrt(v_hat) + EPS)


class Newton:
    """Newton's method: x <- x - H^{-1} g. Uses curvature, so on a quadratic it jumps to the
    stationary point in one step. It heads for ANY stationary point, including saddles,
    because it does not check that H is positive definite."""
    name = "Newton"

    def __init__(self, lr=1.0):
        self.lr = lr

    def reset(self, dim):
        pass

    def step(self, x, prob):
        return x - self.lr * np.linalg.solve(prob.hess(x), prob.grad(x))


class GaussNewton:
    """Gauss-Newton for least squares f = 0.5 ||r(x)||^2.
    Approximates the Hessian by J^T J (drops the second-derivative-of-residual term), so it needs
    only first derivatives of r, and J^T J is always positive semi-definite."""
    name = "Gauss-Newton"

    def __init__(self, lr=1.0):
        self.lr = lr

    def reset(self, dim):
        pass

    def step(self, x, prob):
        J, r = prob.jac(x), prob.residual(x)
        return x - self.lr * np.linalg.solve(J.T @ J, J.T @ r)


def run(opt, prob, x0, iters=5000, tol=1e-8):
    """Run one optimizer under fixed budget/tolerance. Stops early when ||grad|| < tol
    or when the iterate diverges. Returns a dict with the loss history and summary numbers."""
    x = np.array(x0, dtype=float)
    opt.reset(len(x))
    hist = [prob.f(x)]
    path = [x.copy()]
    converged, diverged = False, False
    start = time.perf_counter()
    for k in range(iters):
        x = opt.step(x, prob)
        fx = prob.f(x)
        hist.append(fx)
        path.append(x.copy())
        if not np.isfinite(fx) or abs(fx) > 1e10:
            diverged = True
            break
        if np.linalg.norm(prob.grad(x)) < tol:
            converged = True
            break
    return {
        "x": x,
        "f": hist[-1],
        "hist": np.array(hist),
        "path": np.array(path),
        "iters": len(hist) - 1,
        "time_s": time.perf_counter() - start,
        "converged": converged,
        "diverged": diverged,
    }


FIRST_ORDER = [GD, Momentum, NAG, AdaGrad, RMSProp, Adam]
