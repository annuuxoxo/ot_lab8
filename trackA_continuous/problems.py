"""Data-driven problems for Track A: nonlinear least squares (for Gauss-Newton) and a
finite-sum linear regression (for SGD / mini-batch SGD)."""
import numpy as np


class ExpFit:
    """Fit y = a * exp(b * t) to noisy data. Parameters p = (a, b).
    Loss f(p) = 0.5 * sum r_i^2 with residual r_i = a exp(b t_i) - y_i."""
    name = "expfit"
    x0 = np.array([1.0, 0.0])

    def __init__(self, n=50, a=2.0, b=-1.5, noise=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.t = np.linspace(0, 1, n)
        self.y = a * np.exp(b * self.t) + noise * rng.standard_normal(n)
        self.true = np.array([a, b])

    def residual(self, p):
        a, b = p
        return a * np.exp(b * self.t) - self.y

    def jac(self, p):
        a, b = p
        e = np.exp(b * self.t)
        return np.column_stack([e, a * self.t * e])

    def f(self, p):
        r = self.residual(p)
        return 0.5 * r @ r

    def grad(self, p):
        return self.jac(p).T @ self.residual(p)

    def hess(self, p):
        """Full Hessian = J^T J + sum r_i * Hess(r_i). Used by Newton, for comparison with Gauss-Newton."""
        a, b = p
        J, r = self.jac(p), self.residual(p)
        e = np.exp(b * self.t)
        # second derivatives of r_i: d2/da2 = 0, d2/dadb = t e, d2/db2 = a t^2 e
        H2 = np.array([[0.0, np.sum(r * self.t * e)],
                       [np.sum(r * self.t * e), np.sum(r * a * self.t ** 2 * e)]])
        return J.T @ J + H2


class LinReg:
    """Finite-sum least squares: f(w) = (1/n) sum_i 0.5 (x_i . w - y_i)^2.
    The exact minimizer is known in closed form, so we can measure f - f*."""
    name = "linreg"

    def __init__(self, n=2000, d=20, noise=0.1, seed=0):
        rng = np.random.default_rng(seed)
        # Features with different scales -> moderately ill-conditioned problem
        self.X = rng.standard_normal((n, d)) * np.linspace(0.3, 3.0, d)
        w_true = rng.standard_normal(d)
        self.y = self.X @ w_true + noise * rng.standard_normal(n)
        self.n, self.d = n, d
        self.w_star = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
        self.f_star = self.f(self.w_star)
        # Lipschitz constant of the full gradient = largest eigenvalue of X^T X / n
        self.L = np.linalg.eigvalsh(self.X.T @ self.X / n).max()

    def f(self, w):
        r = self.X @ w - self.y
        return 0.5 * r @ r / self.n

    def grad_batch(self, w, idx):
        """Gradient on a subset of samples (a mini-batch); unbiased estimate of the full gradient."""
        Xb, yb = self.X[idx], self.y[idx]
        return Xb.T @ (Xb @ w - yb) / len(idx)
