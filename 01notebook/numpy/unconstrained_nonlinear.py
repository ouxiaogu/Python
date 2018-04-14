# coding: utf-8
import numpy as np
from sympy import *
import sys
sys.path.append(r'C:\Users\peyang\Perforce\peyang_LT324319_3720\app\mxp\scripts\util')
import logger
logger.initlogging(debug=True)
log = logger.getLogger("UNS")

"""
https://en.wikipedia.org/wiki/Nonlinear_programming
[hide]
- Unconstrained nonlinear: Methods calling ...
... functions
    - Golden-section search *
    - Fibonacci search *
    - Interpolation methods
    - Line search
    - Nelder-Mead method
    - Successive parabolic interpolation
... and gradients
    - Convergence
        - Trust region Wolfe conditions
    - Quasi-Newton
        - BFGS and L-BFGS DFP Symmetric rank-one (SR1)
    - Other methods
        - Berndt-Hall-Hall-Hausman
        - Gauss-Newton
        - Gradient
        - Levenberg-Marquardt
        - Conjugate gradient
        - Truncated Newton
... and Hessians
    Newton's method
"""

def advance_retreat(f, x0, h=0.1):
    """
    For 1D search extreme problem, in condition of infinite variable
    range(no constraints), if the object function f(x) only have 1
    valley, then we can advance and retreat strategy to get the interval
    where minimum locates.
    For more details, refer to https://en.wikipedia.org/wiki/Golden_ratio#Relationship_to_Fibonacci_sequence

    min f(x), x <- R, for x1 <= x2 in [a, b]
    if f(x1) >= f(x2), min locates in [x1, b];
    if f(x1) <  f(x2), min locates in [a, x2];

    Parameters
    ----------
    f : symbol function f(x)
        x = Symbol('x'), y = 2*x**3 - 4*x**2 + x
    x0 : float
        initial guess of x value
    h : float
        initial step length for next step guess

    Returns
    -------
    x_lb : float
        lower bound for optimal
    x_ub : float
        upper bound for optimal
    """
    x = list(f.free_symbols)[0]
    iterator = 0
    x1 = x0
    x2 = x0 + h
    x_ub = x2
    x_lb = x1
    isLtX2 = False

    while True:
        f_x1 = f.subs(x, x1)
        f_x2 = f.subs(x, x2)
        if f_x2 >= f_x1:
            x_ub = x2

            if iterator == 0:
                isLtX2 = True
                h = -h  # search at inverse direction
            if iterator != 0 and  not isLtX2:
                break

            x2 = x1
            x1 = x1 + h

        else: # f_x2 < f_x1
            x_lb = x1
            if iterator != 0 and isLtX2:
                break

            x1 = x2
            x2 = x2 + h

        h = 2*h # speed up convergence
        iterator += 1

        log.debug("advance & retreat, iteration {}, step length {}, bounds [{}, {}]".format(iterator, h, x_lb, x_ub))
    log.info("advance & retreat, final bounds [{}, {}]".format(x_lb, x_ub))
    return x_lb, x_ub

def golden_selection_search(f, a, b, tol=1e-6):
    """
    For 1D search extreme problem, in condition the object function f(x)
    only have 1 valley, if we already narrow down the interval of minimum by
    the "advance and retreat" strategy into [a, b]

    With golden selection search method, let
    x1 = a + 0.382*(b - a)
    x2 = a + 0.618*(b - a)
    if f(x1) >= f(x2), min locates in [x1, b];
    if f(x1) <  f(x2), min locates in [a, x2];

    For more details, refer to https://en.wikipedia.org/wiki/Golden-section_search

    Parameters
    ----------
    f : symbol function f(x)
        e.g., x = Symbol('x'), y = 2*x**3 - 4*x**2 + x
    a : float
        varibale lower bound
    b : float
        varibale upper bound
    tol: float
        solver tolerance, terminate optimization when |x2-x1| < tol

    Returns
    -------
    res : float
        the optimal value found by solver
    """
    x = list(f.free_symbols)[0]
    a, b = (min(a,b), max(a,b) )
    maxIteration = 1e3
    iterator = 0
    gratio   = 0.618
    while iterator < maxIteration:
        x1 = a + (1 - gratio)*(b - a)
        x2 = a + gratio*(b - a)
        if abs(x2 - x1) < tol:
            break

        f_x1 = f.subs(x, x1)
        f_x2 = f.subs(x, x2)

        if f_x1 <= f_x2:
            b = x2
        else:
            a = x1
        iterator += 1
        log.debug("golden selection search, iteration {}, bounds [{}, {}]".format(iterator, a, b))
    if iterator == maxIteration:
        log.info("golden selection search reach max iteration number {}, failed to converge", maxIteration)
    res = (a + b)/2.
    log.info("golden selection search, final result {}: {}".format(res, f.subs(x, res)) )
    return res

def fibonacci_search(f, a, b, tol=1e-6):
    """
    For 1D search extreme problem, in condition the object function f(x)
    only have 1 valley, if we already narrow down the interval of minimum by
    the "advance and retreat" strategy into [a, b]

    In Fibonacci search method,

    Firstly, evaluate the fibonacci number Fn by:
        F(n) >= (b - a)/tol

    x1_i = a + F(i-2)/F(i)*(b - a)
    x2_i = a + F(i-1)/F(i)*(b - a)
    if f(x1) >= f(x2), min locates in [x1, b];
    if f(x1) <  f(x2), min locates in [a, x2];

    For more details, refer to https://en.wikipedia.org/wiki/Golden-section_search

    Parameters
    ----------
    f : symbol function f(x)
        e.g., x = Symbol('x'), y = 2*x**3 - 4*x**2 + x
    a : float
        varibale lower bound
    b : float
        varibale upper bound
    tol : float
        solver tolerance, terminate optimization when |x2-x1| < tol

    Returns
    -------
    res : float
        the optimal value found by solver

    Reference
    ---------
    https://en.wikipedia.org/wiki/Golden_ratio#Relationship_to_Fibonacci_sequence
    Phi^k = F(k)*Phi + F(k-1), Mathematical Induction can prove
    """
    x = list(f.free_symbols)[0]
    a, b = (min(a, b), max(a, b))
    tol = abs(tol)
    bound = (b - a)/tol
    fibs = fibonacci_array(bound)
    print fibs, " ", len(fibs)

    maxIteration = len(fibs)
    iterator = 0
    while iterator < maxIteration - 2:
        # reverse direction using fibs
        x1 = a + 1.0 * fibs[maxIteration-1 - iterator - 2]/fibs[maxIteration-1 - iterator] * (b - a)
        x2 = a + 1.0 * fibs[maxIteration-1 - iterator - 1]/fibs[maxIteration-1 - iterator] * (b - a)
        # positive direction
        # x1 = a + 1. * fibs[iterator]/fibs[iterator + 2] * (b - a)
        # x2 = a + 1. * fibs[iterator + 1]/fibs[iterator + 2] * (b - a)
        if abs(x2 - x1) < tol:
            break

        f_x1 = f.subs(x, x1)
        f_x2 = f.subs(x, x2)

        if f_x1 <= f_x2:
            b = x2
        else:
            a = x1
        iterator += 1
        log.debug("Fibonacci search, iteration {}, bounds [{}, {}]".format(iterator, a, b))
    res = (a + b)/2.
    log.info("Fibonacci search, final result {}: {}".format(res, f.subs(x, res)) )
    return res

def fibonacci_search_reuse(f, a, b, tol=1e-6):
    """
    For 1D search extreme problem, in condition the object function f(x)
    only have 1 valley, if we already narrow down the interval of minimum by
    the "advance and retreat" strategy into [a, b]

    In Fibonacci search method,

    Firstly, evaluate the fibonacci number Fn by:
        F(n) >= (b - a)/tol

    x1_i = a + F(i-2)/F(i)*(b - a)
    x2_i = a + F(i-1)/F(i)*(b - a)
    if f(x1) >= f(x2), min locates in [x1_i, b], reuse x2_i as x1_(i+1), recalcaluate x2_(i+1) = a' + (b'-a')*F(k-1)/F(k)
    if f(x1) <  f(x2), min locates in [a, x2_i], reuse x1_i as x2_(i+1), recalcaluate x1_(i+1) = a' + (b'-a')*F(k-2)/F(k)

    For more details, refer to https://en.wikipedia.org/wiki/Golden-section_search

    Parameters
    ----------
    f : symbol function f(x)
        e.g., x = Symbol('x'), y = 2*x**3 - 4*x**2 + x
    a : float
        varibale lower bound
    b : float
        varibale upper bound
    tol : float
        solver tolerance, terminate optimization when |x2-x1| < tol

    Returns
    -------
    res : float
        the optimal value found by solver
    """
    x = list(f.free_symbols)[0]
    a, b = (min(a, b), max(a, b))
    tol = abs(tol)
    bound = (b - a)/tol
    fibs = fibonacci_array(bound)
    print fibs, " ", len(fibs)

    maxIteration = len(fibs)
    # initialization
    iterator = 0
    x1 = a + 1.0 * fibs[maxIteration-1 - iterator - 2]/fibs[maxIteration-1 - iterator] * (b - a)
    x2 = a + 1.0 * fibs[maxIteration-1 - iterator - 1]/fibs[maxIteration-1 - iterator] * (b - a)
    while iterator < maxIteration - 2:
        if abs(x2 - x1) < tol:
            break

        f_x1 = f.subs(x, x1)
        f_x2 = f.subs(x, x2)

        if f_x1 <= f_x2:
            b = x2
            x2 = x1
            x1 = a + 1.0 * fibs[maxIteration-1 - iterator - 2]/fibs[maxIteration-1 - iterator] * (b - a)
        else:
            a = x1
            x1 = x2
            x2 = a + 1.0 * fibs[maxIteration-1 - iterator - 1]/fibs[maxIteration-1 - iterator] * (b - a)
        iterator += 1
        log.debug("Fibonacci search reuse, iteration {}, bounds [{}, {}]".format(iterator, a, b))
    res = (a + b)/2.
    log.info("Fibonacci search reuse, final result {}: {}".format(res, f.subs(x, res)) )
    return res

def fibonacci_array(bound):
    """
    Return the fibonacci array util F(n) >= bound

    Parameters
    ----------
    bound : float
        The inequality here: F(n) >= bound

    Returns
    -------
    fibs : list
        fibonacci array [F(0), F(1), ..., F(n)], size of n+1
    """
    if bound < 0:
        return []
    elif bound < 1:
        return [0]
    fibs = [0, 1]
    while fibs[-1] < bound:
        fibs.extend([ fibs[-2] + fibs[-1] ])
    return fibs[2:]

def newton_search(f, x0, stepTol=1e-6, rmsTol=1e-6):
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

    Limitation:
    Just find the local minimum nearest to the start point,
    failed when it is a maxima near start print

    For more details, refer to https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization

    Parameters
    ----------
    f : symbol function f(x)
        e.g., x = Symbol('x'), y = 2*x**3 - 4*x**2 + x
    x0 : float
        initial guess
    stepTol : float
        minimum movement value allowed, |x2-x1| < stepTol
    rmsTol : float
        minimum movement value allowed, |rms2-rms1| < rmsTol

    Returns
    -------
    res : float
        the optimal value found by solver
    """
    x = list(f.free_symbols)[0]
    df = f.diff(x)
    d2f = df.diff(x)

    maxIteration = 100
    iterator = 0
    solverFailed = False
    preRms = f.subs(x, x0) + 1000
    eps = stepTol
    while iterator < maxIteration:
        f_x0 = f.subs(x, x0)
        df_x0 = df.subs(x, x0)
        d2f_x0 = d2f.subs(x, x0)

        log.debug("Newton's method search, iteration {}, x: {}, f(x): {}".format(iterator, x0, f_x0))

        if d2f_x0 == 0 :
            log.info("Newton's method search, failed to converge: check d2f {} ".format(d2f_x0))
            solverFailed = True
            break
        if f_x0 > preRms: # uncomment this part may converge to a local maxima
            log.info("Newton's method search, failed to converge: check rms change: {} -> {}".format(preRms, f_x0))
            solverFailed = True
            break

        if abs(df_x0) < eps:
            log.debug("Newton's method search, rms converged: df_x0 -> {}".format(df_x0))
            break
        x1 = x0 - 1.0*df_x0/d2f_x0
        # if abs(x1 - x0) < stepTol or abs(f_x0 - preRms) < rmsTol:
        #     log.debug("Newton's method search, converged check step: {} -> {}, or rms: {} -> {}".format(x0, x1, preRms, f_x0))
        #     break

        x0 = x1
        iterator += 1
    res = x0
    log.info("Newton's method search, final result {}: {}".format(res, f.subs(x, res)) )
    return res

def newton_search_global(f, x0, stepTol=1e-6, rmsTol=1e-6):
    """
    How to overcome the Limitation of Newton search method:
    "Just find the local minimum nearest to the start point,
        failed when it is a maxima near start print"
    What about climbing over the local maxima to reach the minima close to it

    For more details, refer to https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization

    Parameters
    ----------
    f : symbol function f(x)
        e.g., x = Symbol('x'), y = 2*x**3 - 4*x**2 + x
    x0 : float
        initial guess
    stepTol : float
        minimum movement value allowed, |x2-x1| < stepTol
    rmsTol : float
        minimum movement value allowed, |rms2-rms1| < rmsTol

    Returns
    -------
    res : float
        the optimal value found by solver
    """
    x = list(f.free_symbols)[0]
    df = f.diff(x)
    d2f = df.diff(x)

    maxIteration = 100
    iterator = 0
    solverFailed = False
    preRms = f.subs(x, x0) + 1000
    while iterator < maxIteration:
        f_x0 = f.subs(x, x0)
        df_x0 = df.subs(x, x0)
        d2f_x0 = d2f.subs(x, x0)

        log.debug("Newton's method search, iteration {}, x: {}, f(x): {}".format(iterator, x0, f_x0))

        if abs(f_x0 - preRms) < rmsTol:
            log.debug("Newton's method search, rms converged: {} -> {}".format(preRms, f_x0))
            break

        '''
        Change 1 for global Newton method:
        Don't terminate at zero 1st derivative, but check the sign of 2nd
        derivative.  Fr f'(x) = 0, if f''(x0) >= 0, yes, it's minima, stop

        '''
        if abs(d2f_x0) < 1e-10:
            log.info("Newton's method search, failed to converge: check d2f {} ".format(d2f_x0, preRms, f_x0))
            if f_x0 > preRms: # climb over
                maxInnerIters = 100
                innerIter = 0
                delta = stepTol
                while innerIter < maxInnerIters and f_x0 > preRms:
                    x0 += delta
                    delta += delta
                    innerIter += 1

        x1 = x0 - 1.0*df_x0/d2f_x0
        if abs(x1 - x0) < stepTol:
            log.debug("Newton's method search, step converged: {} -> {}".format(x0, x1))
            break

        x0 = x1
        iterator += 1
    res = x0
    log.info("Newton's method search, final result {}: {}".format(res, f.subs(x, res)) )
    return res

if __name__ == '__main__':
    log.info("Unconstrained nonlinear solvers with known analysis formula")
    x = symbols('x')
    f = x**4 - x**2 - 2*x + 5

    """# advance and retreat
                vlb, vub = advance_retreat(f, 0, 0.1)
                golden_selection_search(f, vlb, vub, 1e-3)
                import time
                start_time = time.time()
                fibonacci_search(f, vlb, vub, 1e-3)
                log.debug("Fibonacci search, elapsed time {}".format(time.time() - start_time))
                start_time = time.time()
                fibonacci_search_reuse(f, vlb, vub, 1e-3)
                log.debug("Fibonacci search reuse, elapsed time {}".format(time.time() - start_time))"""

    f = x**3 - 3*x + 2
    newton_search(f, -100) # failed to converge case, start at left of a maxima
    # newton_search(f, 100)