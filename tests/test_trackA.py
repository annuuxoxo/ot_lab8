import os
import sys

import numpy as np
import pytest
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trackA_continuous"))
from functions import ALL, Quadratic  # noqa: E402
from optimizers import FIRST_ORDER, GaussNewton, Newton, run  # noqa: E402
from problems import ExpFit, LinReg  # noqa: E402


def fd_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


@pytest.mark.parametrize("prob", ALL + [ExpFit()], ids=lambda p: p.name)
def test_gradient_matches_finite_differences(prob):
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = rng.uniform(-1.5, 1.5, 2)
        assert np.allclose(prob.grad(x), fd_grad(prob.f, x), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("prob", ALL + [ExpFit()], ids=lambda p: p.name)
def test_hessian_matches_finite_differences(prob):
    x = np.array([0.7, -0.4])
    H_fd = np.column_stack([fd_grad(lambda z, i=i: prob.grad(z)[i], x) for i in range(2)]).T
    assert np.allclose(prob.hess(x), H_fd, rtol=1e-4, atol=1e-4)


def test_newton_solves_quadratic_in_one_step():
    q = Quadratic()
    r = run(Newton(), q, q.x0)
    assert r["iters"] == 1 and np.allclose(r["x"], 0)


def test_newton_is_attracted_to_saddle():
    """Newton ignores whether H is positive definite, so near the ridge it converges to the saddle (0,0)."""
    from functions import Saddle
    s = Saddle()
    r = run(Newton(), s, s.x0)
    assert np.allclose(r["x"], 0, atol=1e-6)


@pytest.mark.parametrize("cls", FIRST_ORDER, ids=lambda c: c.name)
def test_first_order_methods_minimize_quadratic(cls):
    q = Quadratic()
    lr = 0.01 if cls.name in ("GD", "Momentum", "NAG") else 0.05
    r = run(cls(lr), q, q.x0, iters=20000, tol=1e-6)
    assert q.f(r["x"]) < 1e-6


def test_gauss_newton_matches_scipy():
    p = ExpFit()
    ref = least_squares(p.residual, p.x0, jac=p.jac, xtol=1e-14, ftol=1e-14, gtol=1e-14).x
    r = run(GaussNewton(), p, p.x0, iters=100)
    assert np.allclose(r["x"], ref, atol=1e-6)


def test_minibatch_gradient_is_full_gradient_when_batch_is_everything():
    lr = LinReg(n=200, d=5)
    w = np.ones(5)
    full = lr.X.T @ (lr.X @ w - lr.y) / lr.n
    assert np.allclose(lr.grad_batch(w, np.arange(lr.n)), full)
    assert np.allclose(fd_grad(lr.f, w), full, atol=1e-5)
