# coding: utf-8
import numpy as np
from sympy import Matrix, symbols, pprint

import sys
import os.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../common")
from logger import logger
log = logger.getLogger(__name__)

"""
Consider a set of m data points, and a curve (model function) y=f(x),
x depends on n parameters, x=[x1, x2, ..., xn], with m>=n,
the measured value is y = [y1, y2, .., ym], the problem to solve here is
S = \sum _i (y_i - f(x_i))^2, min{S}

https://en.wikipedia.org/wiki/Non-linear_least_squares

Algorithms including but not limited to:
4.1 Gauss–Newton method
4.1.1   Shift-cutting
4.1.2   Marquardt parameter
4.2 QR decomposition
4.3 Singular value decomposition
4.4 Gradient methods
4.5 Direct search methods
"""

def gaussian_newton_method(f, syms, beta0, stepTol=1e-6):
    """
    Newton search use the information of target function itself, to find
    a very direct optimization path.

    Taylor expand:
    f(x0 + dx) ~= f(x0) + f'(x0)*dx + f''(x0)/2! *(dx)^2
    To solve the minima of f(x0 + dx) regarding dx, extreme value
    at stational points(驻点), with 1st derivative equal to zero.
    so we have \delta f(x0+dx) / \delta dx = 0 => dx = f'(x0)/f''(x0)

    The newton search guess is:

        x_{n+1} = x_{n} - f'(x_{n})/f''(x_{n})

    Based on the Newton method, but to min{S(X)}, we have
        m observations, recorded as residual, r = [r_1, r_2, ..., r_m], m*1
        n variables, recorded as solution, beta = [x_1, x_2, .., x_n], n*1
        J = [J_i,j], J_i,j = d1r_i(beta_j) // m*n
        S = [r_i^2] // m*1

    and here
        S'(X)  = 2f'(X)*f(X) = 2*Jf // n*1 = n*m * m*1
        S''(X) = 2f''(X)*f(X) + 2f'(X)*f'(X) ~= 2f'(X)*f'(X) = 2*JTJ  // n*n = n*m * m*n

    Assumption:
        1. {x| S(x) <= S(x0)} is bounded 有界
        2. Jacobian matrix J(x) has full rank in all steps 矩阵每一步都是满秩
        3. Need to meet:
            f''(X)*f(x) << f'(x)*f'(x)
            r_i * d2r_i(beta_j, beta_k) << d1r_i(beta_j) * d1r_i(beta_k)

    Limitation:
        1. 一种局部收敛法，对初始点依赖很大, 只有当初始点接近极小值点时才可能收敛

    For more details, refer to https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm

    Parameters
    ----------
    f : symbol function Matrix
        e.g., f(x) = [x+1,  y+x-1], will convert to f(x) = (x+1)^2 + (y+x-1)^2
    syms: iterable
        n variables in iterable symbols format
    x0 : iterable
        initial guess, same order with syms
    stepTol : float
        minimum movement value allowed, |x1 - x0| < stepTol

    Returns
    -------
    res : float
        the optimal value found by solver
    """
    from collections import Iterable
    if not isinstance(x0, Iterable):
        x0 = (x0, )
    if not isinstance(syms, Iterable):
        syms = (syms, )
    syms = Matrix(syms)
    x0 = Matrix(x0)
    assert(len(x0) == len(syms))
    S = f.T*f
    J = f.jacobian(syms) # m*n
    JTJ = J.T*J # n*n
    Jf = J.T*f # n*1

    maxIteration = 100
    iterator = 0
    solverFailed = False
    eps = stepTol
    while True:
        kvargs = {k: x0[i] for i, k in enumerate(syms)}
        S_x0 = S.subs(kvargs)

        log.debug("Gaussian-Newton method search, iteration {}, x: {}, S(x): {}".format(iterator, x0, S_x0))

        JTJ_x0 = JTJ.subs(kvargs)
        Jf_x0 = Jf.subs(kvargs)
        if abs(JTJ_x0.det() ) < eps :
            log.info("Gaussian-Newton method search, failed to converge: check JTJ_x0 {} ".format(str(JTJ_x0)) )
            solverFailed = True
            break

        dx = - JTJ_x0**(-1) * Jf_x0

        if dx.norm() < eps:
            log.debug("Gaussian-Newton method search, rms converged: dx.norm() -> {}".format(dx.norm() ) )
            break
        x1 = x0 + dx

        iterator += 1
        if iterator > maxIteration:
            break
        x0 = x1

    res = x0
    kvargs = {k: res[i] for i, k in enumerate(syms)}
    log.info("Gaussian-Newton method search, final result {}: {}".format(res, S.subs(kvargs)) )
    return res

def modified_gaussian_newton_method(f, syms, x0, stepTol=1e-6):
    """
    Based on the classical Gaussion-Newton method, but to min{S(X)}, we have

        S'(X)  = 2f'(X)*f(X) = 2 * J.T *f // n*1 = n*m * m*1
        S''(X) = 2f''(X)*f(X) + 2f'(X)*f'(X) ~= 2f'(X)*f'(X) = 2*J.T * J  // n*n = n*m * m*n

    To overcome the limitation, methods:
        1. When to get a minima guess x_k, we can use 1D unconstrained nonlinear method to search the minima at direction of -JTJ^(-1) * Jf, get the a optimal α to make x_{k+1} = x_k - α * JTJ^(-1) * Jf, has the minimum cost # but this method is costly, O(n^2) search
        2. at direction of V = -JTJ^(-1) * Jf, choose a small positive delta dx, let S(x_k + dx) < S(x_k), dx = α * V.

    Here we follow method 2, and the flow to find α is:
        - let α = 1,  β = 1e-5, dx = -JTJ^(-1) * Jf
        - if S(x_k + α * dx) <= S(x_k) + 2*β * α * dx.T * Jf
                            = S(x_k) - 2*β * α * (Jf).T * JTJ^(-1).T * Jf
                            = S(x_k) - 2*β * α * |det| * (Jf).T * Jf
            then continue GN
        - else, α = 0.5 * α util S(x_k + α * dx) <= S(x_k) + β * α * dx.T * Jf

        因为 JTJ^(-1).T 是J^(-2), 满秩时总归可以写成 |det|*I, 所以 |det| * (Jf).T * Jf 为一个正数，系数为-β * α 也就是一个很小的负数，因此 满足 S(x_k + dx) < S(x_k)

    For more details, refer to https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm

    Parameters
    ----------
    f : symbol function Matrix
        e.g., f(x) = [x+1,  y+x-1], will convert to f(x) = (x+1)^2 + (y+x-1)^2
    syms: iterable
        n variables in iterable symbols format
    x0 : iterable
        initial guess, same order with syms
    stepTol : float
        minimum movement value allowed, |x1 - x0| < stepTol

    Returns
    -------
    res : float
        the optimal value found by solver
    """
    from collections import Iterable
    if not isinstance(x0, Iterable):
        x0 = (x0, )
    if not isinstance(syms, Iterable):
        syms = (syms, )
    syms = Matrix(syms)
    x0 = Matrix(x0)
    assert(len(x0) == len(syms))
    S = f.T*f
    J = f.jacobian(syms) # m*n
    JTJ = J.T*J # n*n
    Jf = J.T*f # n*1

    maxIteration = 100
    iterator = 0
    solverFailed = False
    eps = stepTol
    beta = 1e-5

    while True:

        kvargs = {k: x0[i] for i, k in enumerate(syms)}
        S_x0 = S.subs(kvargs)

        log.debug("Modified Gaussian-Newton method search, iteration {}, x: {}, S(x): {}".format(iterator, x0, S_x0))

        JTJ_x0 = JTJ.subs(kvargs)
        Jf_x0 = Jf.subs(kvargs)
        if abs(JTJ_x0.det() ) < eps :
            log.info("Modified Gaussian-Newton method search, failed to converge: check JTJ_x0 {} ".format(str(JTJ_x0)) )
            solverFailed = True
            break

        dx = - JTJ_x0**(-1) * Jf_x0
        alpha = 1
        while True:
            kvargs = {k: x0[i] + alpha*dx[i] for i, k in enumerate(syms)}
            S1 = S.subs(kvargs)
            S2 = S_x0 + 2 * beta * alpha * dx.T * Jf_x0 # β * α * dx.T * Jf
            if S1 < S2:
                break
            alpha *= 0.5
        if dx.norm() < eps:
            log.debug("Modified Gaussian-Newton method search, rms converged: dx.norm() -> {}".format(dx.norm() ) )
            break
        x1 = x0 + alpha*dx
        iterator += 1
        if iterator > maxIteration:
            break
        x0 = x1

    res = x0
    kvargs = {k: res[i] for i, k in enumerate(syms)}
    log.info("Modified Gaussian-Newton method search, final result {}: {}".format(res, S.subs(kvargs)) )
    return res

def LM_method(f, syms, x0, stepTol=1e-6):
    """
    Based on the classical Levenberg-Marquardt method, 相当于 Gaussian-Newton method的修正

        S的一阶导数，或梯度向量方向
        S'(X) = 2 * J.T *f // n*1 = n*m * m*1
        GN method, 即为梯度向量下降的方向
        δ = - J.T *f / (J.T×J)

        LM method的修正为：
        (J.T*J + λ*I)*δ = J.T * f

        λ 分离成 u, v来调整，u是初始常值，v是初始系数项(v>1)，λ0 = u

        1.（非负）阻尼系数λ在每次迭代时都会被调整。
        2. 如果S的减少速度很快，则可以使用较小的λ，使算法更接近GN method
            If S(x0 + δ) < S(x0) + β*(J.T * f).T * δ, then λ = λ/v
        3. 如果迭代不足以减少残差，则可以增加λ，giving a step closer to the gradient-descent direction。

    Here we follow method 2, and the flow to find α is:
        - let α = 1,  β = 1e-5, dx = -JTJ^(-1) * Jf
        - if S(x_k + α * dx) <= S(x_k) + 2*β * α * dx.T * Jf
                            = S(x_k) - 2*β * α * (Jf).T * JTJ^(-1).T * Jf
                            = S(x_k) - 2*β * α * |det| * (Jf).T * Jf
            then continue GN
        - else, α = 0.5 * α util S(x_k + α * dx) <= S(x_k) + β * α * dx.T * Jf

        因为 JTJ^(-1).T 是J^(-2), 满秩时总归可以写成 |det|*I, 所以 |det| * (Jf).T * Jf 为一个正数，系数为-β * α 也就是一个很小的负数，因此 满足 S(x_k + dx) < S(x_k)

    For more details, refer to https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm

    Parameters
    ----------
    f : symbol function Matrix
        e.g., f(x) = [x+1,  y+x-1], will convert to f(x) = (x+1)^2 + (y+x-1)^2
    syms: iterable
        n variables in iterable symbols format
    x0 : iterable
        initial guess, same order with syms
    stepTol : float
        minimum movement value allowed, |x1 - x0| < stepTol

    Returns
    -------
    res : float
        the optimal value found by solver
    """
    from collections import Iterable
    if not isinstance(x0, Iterable):
        x0 = (x0, )
    if not isinstance(syms, Iterable):
        syms = (syms, )
    syms = Matrix(syms)
    x0 = Matrix(x0)
    assert(len(x0) == len(syms))
    S = f.T*f
    J = f.jacobian(syms) # m*n
    JTJ = J.T*J # n*n
    Jf = J.T*f # n*1

    maxIteration = 100
    iterator = 0
    solverFailed = False
    eps = stepTol
    beta = 1e-5

    while True:

        kvargs = {k: x0[i] for i, k in enumerate(syms)}
        S_x0 = S.subs(kvargs)

        log.debug("Modified Gaussian-Newton method search, iteration {}, x: {}, S(x): {}".format(iterator, x0, S_x0))

        JTJ_x0 = JTJ.subs(kvargs)
        Jf_x0 = Jf.subs(kvargs)
        if abs(JTJ_x0.det() ) < eps :
            log.info("Modified Gaussian-Newton method search, failed to converge: check JTJ_x0 {} ".format(str(JTJ_x0)) )
            solverFailed = True
            break

        dx = - JTJ_x0**(-1) * Jf_x0
        alpha = 1
        while True:
            kvargs = {k: x0[i] + alpha*dx[i] for i, k in enumerate(syms)}
            S1 = S.subs(kvargs)
            S2 = S_x0 + 2 * beta * alpha * dx.T * Jf_x0 # β * α * dx.T * Jf
            if S1 < S2:
                break
            alpha *= 0.5
        if dx.norm() < eps:
            log.debug("Modified Gaussian-Newton method search, rms converged: dx.norm() -> {}".format(dx.norm() ) )
            break
        x1 = x0 + alpha*dx
        iterator += 1
        if iterator > maxIteration:
            break
        x0 = x1

    res = x0
    kvargs = {k: res[i] for i, k in enumerate(syms)}
    log.info("Modified Gaussian-Newton method search, final result {}: {}".format(res, S.subs(kvargs)) )
    return res

def Df(exprs, symbols):
    """
    Df matrix: Jacobian with known analytic expressions

    Jacobian Matrix:
        m observations, recorded as residual, r = [r_1, r_2, ..., r_m], m*1
        n variables, recorded as solution, beta = [x_1, x_2, .., x_n], n*1
        J = [J_i,j], J_i,j = d1r_i(beta_j) // m*n
    Df Matrix:
        m known analytic expressions, r = [r_1, r_2, ..., r_m], m*1
    Need to have m >= n

    parameters
    ----------
    exps: Matrix
        m*1, analytic expressionsf
    symbols: cell
        n*1
    """
    return exprs.jacobian(symbols)


if __name__ == '__main__':

    """ pretty printing
    from sympy import Integral, sqrt
    x = symbols('x')
    pprint(Integral(sqrt(1/x), x), use_unicode=False)
    """

    """
    from sympy import sin, cos
    w, y, z = symbols('w y z')
    x = (w, y)
    A = Matrix([cos(w) - sin(y)**2, 1+sin(y)**2, 1+sin(y)**2 + z])
    print Df(A, x)
    print A.free_symbols
    print A.transpose()*A
    pprint(A.T*A)
    """

    t = symbols('t') # n=1
    f = Matrix([t+1, 0.5*t**2+t-1]) # m=2
    # """
    S = f.T*f
    pprint(S)
    S_x0 = S.subs((t,), (1,) )
    print("S_x0 wrong="); pprint(S_x0)
    kvargs ={t: 1}
    S_x0 = S.subs(t, 1)
    S_x0 = S.subs(kvargs)
    print("S_x0="); pprint(S_x0)
    J = f.jacobian((t,) )
    print("J=");pprint(J)
    J_x0 = J.subs(kvargs)
    print("J_x0 = "); pprint(J_x0)
    print("J.norm = "); pprint(J.norm() )
    print("J_x0.norm = "); pprint(J_x0.norm() )
    JTJ = J.T * J
    print("JTJ = "); pprint(JTJ)
    JTJ_x0 = JTJ.subs(kvargs)
    print("JTJ_x0 = "); pprint(JTJ_x0)
    print("JTJ_x0.shape = "); pprint(JTJ_x0.shape)

    # """
    # gaussian_newton_method(f, t, 1)

    # t = symbols('t') # n=1
    # f = Matrix([t+1, t**2+t+1]) # m=2
    # modified_gaussian_newton_method(f, t, 1)
