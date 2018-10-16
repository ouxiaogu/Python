# coding: utf-8
import numpy as np
from sympy import *
import sys
sys.path.insert(0, r'C:\Users\peyang\Perforce\peyang_LT324319_3720\app\mxp\scripts\util')
import logger
logger.initlogging(debug=True)
log = logger.getLogger("UNS")

"""
Optimization problem is to find the extreme value for f(x)
Here we asume to the find minima

https://en.wikipedia.org/wiki/Nonlinear_programming
[hide]
- Unconstrained nonlinear: Methods calling ...
... functions
    - Line search:
        - Golden-section search *
        - Fibonacci search *
        - Quadratic Interpolation
        - other Interpolation methods: Powell's method
    - Nelder-Mead method
    - Successive parabolic interpolation
... and gradients
    - Convergence
        - Trust region Wolfe conditions.
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
        # if f_x0 > preRms: # uncomment this part may converge to a local maxima
        #     log.info("Newton's method search, failed to converge: check rms change: {} -> {}".format(preRms, f_x0))
        #     solverFailed = True
        #     break

        if abs(df_x0) < eps:
            log.debug("Newton's method search, rms converged: df_x0 -> {}".format(df_x0))
            break
        x1 = x0 - 1.0*df_x0/d2f_x0
        # if abs(x1 - x0) < stepTol or abs(f_x0 - preRms) < rmsTol:
        #     log.debug("Newton's method search, converged check step: {} -> {}, or rms: {} -> {}".format(x0, x1, preRms, f_x0))
        #     break

        x0 = x1
        preRms = f_x0
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
    foundOneExtreme = False
    while iterator < maxIteration:
        f_x0 = f.subs(x, x0)
        df_x0 = df.subs(x, x0)
        d2f_x0 = d2f.subs(x, x0)

        log.debug("Newton's method global search, iteration {}, x: {}, f(x): {}".format(iterator, x0, f_x0))

        '''
        Change 1 for global Newton method:
        Don't terminate at zero 1st derivative, but check the sign of 2nd
        derivative.  Fr f'(x) = 0, if f''(x0) >= 0, yes, it's minima, stop

        '''
        if abs(df_x0) < 1e-10:
            foundOneExtreme = True
            if d2f_x0 >= 0:
                log.debug("Newton's method global search, converged: check f_x0 {} df_x0 {}, d2f_x0 {} ".format(f_x0, df_x0, d2f_x0))
                break
            else:
                log.debug("Newton's method global search, maxima found, need to climb over the maxima, and search at the other side")
                deltaStep = stepTol
                x1 = x0 + deltaStep
                f_x1 = f.subs(x, x1)
                while f_x1 >= f_x0: # climb over
                    deltaStep = 2*deltaStep
                    x1 = x0 + deltaStep
                    f_x1 = f.sub(x, x1)
        else:
            '''
            Change 2 for global Newton method:
                1. reverse sign for d2f_x0 if toward maxima
                2. but if start point at leftest ascending side, need to climp over
            '''

            if foundOneExtreme:
                d2f_x0 = abs(d2f_x0)
            x1 = x0 - 1.0 * df_x0 / d2f_x0


        x0 = x1
        iterator += 1

    res = x0
    log.info("Newton's method global search, final result {}: {}".format(res, f.subs(x, res)) )
    return res

def secant_method(f, x0, x1, eps=1e-6):
    """
    secant method(割线法) use the information of target function, with previous two steps to the postion guess for current step.

    General secant method is built on f(x) to find the root for f(x)=0, for
    optimization problem, just change to find f'(x)=0, then the secant search
    guess is:

        x_{n+1} = x_{n} - f'(x_{n})/[( f'(x_{n}) - f'(x_{n-1}))/(x_{n} - x_{n-1})]

    compared with the newton search guess:

        x_{n+1} = x_{n} - f'(x_{n})/f''(x_{n})

    We can find that, it's just an approximation for f''(x)

        f''(x) ~= (f'(x_{n}) - f'(x_{n-1}))/(x_{n} - x_{n-1})

    Limitation:
    if start point at leftest ascending side, need to climp over
    Just find the local minimum nearest to the start point,
    failed when it is a maxima near start print

    For more details, refer to https://en.wikipedia.org/wiki/Secant_method

    Parameters
    ----------
    f : symbol function f(x)
        e.g., x = Symbol('x'), y = 2*x**3 - 4*x**2 + x
    x0 : float
        start point 1
    x1 : float
        start point 2
    eps : float
        epsilon, terminate at abs( f'(x) ) < epsilon

    Returns
    -------
    res : float
        the optimal value found by solver
    """
    x = list(f.free_symbols)[0]
    df = f.diff(x)

    maxIteration = 100
    iterator = 0
    solverFailed = False
    res = x0

    df_x0 = df.subs(x, x0)
    df_x1 = df.subs(x, x1)
    if abs(df_x0) < eps:
        log.debug("secant method search, converged, check df_x0 -> {}".format(df_x0))
        res = x0
    elif abs(df_x1) < eps:
        log.debug("secant method search, converged, check df_x1 -> {}".format(df_x1))
        res = x1
    else:
        while iterator < maxIteration:
            f_x0 = f.subs(x, x0)
            f_x1 = f.subs(x, x1)
            df_x0 = df.subs(x, x0)
            df_x1 = df.subs(x, x1)

            log.debug("secant method search, iteration {}, x0 {}, f(x0) {}; x1 {}, f(x1) {}".format(iterator, x0, f_x0, x1, f_x1))

            if abs(df_x1) < eps:
                log.debug("secant method search, converged, check df_x1 -> {}".format(df_x1))
                break
            if abs(df_x1 - df_x0) == 0 :
                log.debug("secant method search, failed to converge: check df_x0 {} df_x1 {} ".format(df_x0, df_x1))
                solverFailed = True
                break
            x2 = x1 - 1.0*(x1 - x0)*df_x1/(df_x1 - df_x0)

            x0 = x1
            x1 = x2
            iterator += 1

        res = x1
    log.info("secant method search, final result {}: {}".format(res, f.subs(x, res)) )
    return res

def quadratic_interpolation(f, x0, x1, eps=1e-6):
    """
    quadratic interpolation method(二次插值法) use the x0, x1, t to settle a quadratic curve f(x)=a*x^2+b*x+c, then minima at stational point of quadratic curve, x=-b/(2a).

    The quadratic interpolation method search guess is:

        t' = 1/2 * { [f(x0)*(x1^2 - t^2) - f(t)(x1^2 - x0^2) + f(x1)*(t^2 - x0^2)] / [f(x0)*(x1 - t) - f(t)(x1 - x0) + f(x1)*(t - x)] }

    Limitation:
    if there are leftest ascending curve within range, t will be trapped at this range, can't see any other extreme points. High resolution, but low speed.

    For more details, refer to https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-252j-nonlinear-programming-spring-2003/lecture-notes/

    Parameters
    ----------
    f : symbol function f(x)
        e.g., x = Symbol('x'), y = 2*x**3 - 4*x**2 + x
    x0 : float
        start point 1
    x1 : float
        start point 2
    lambda: float, (0, 1)
        if lambda is None, set as 0.618;
    eps : float
        epsilon, terminate at abs( t_{n+1} - t_{n} ) < epsilon

    Returns
    -------
    res : float
        the optimal value found by solver
    """
    ratio = 0.618
    t = x0 + ratio * (x1 - x0)
    nrepeat = 0

    maxIteration = 100
    iterator = 0
    solverFailed = False
    res = x0

    while iterator < maxIteration:
        f_x0 = f.subs(x, x0)
        f_x1 = f.subs(x, x1)
        f_t = f.subs(x, t)
        log.debug("quadratic interpolation method search, iteration {}, t {}, f(t) {}".format(iterator, t, f_t))

        numerator = f_x0*(x1**2 - t**2) - f_t*(x1**2 - x0**2) + f_x1*(t**2 - x0**2)
        denominator = f_x0*(x1 - t) - f_t*(x1 - x0) + f_x1*(t - x0)
        if denominator == 0:
            log.debug("quadratic interpolation method search, failed to converge because of zero devision error")
            solverFailed = True
            break

        nt = 0.5 * numerator / denominator # guess for next t

        if nrepeat == 2:
            log.debug("quadratic interpolation method search, failed to converge, please check if there is leftest ascending curve within the range")
            solverFailed = True
            break

        f_nt = f.subs(x, nt)
        if abs(t - nt) < eps:
            log.debug("quadratic interpolation method search, guess converged, check step: {} -> {}".format(t, nt))
            if f_nt <= f_t:
                res = nt
            else:
                res = t
            break
        # update search range just like advance and retreat
        if nt < t:
            if f_nt <= f_t:
                x1 = t
                t = nt
                nrepeat = 0
            else:
                x0 = nt
                nrepeat += 1
        else:
            if f_nt <= f_t:
                x0 = t
                t = nt
                nrepeat = 0
            else:
                x1 = nt
                nrepeat += 1
        iterator += 1

    log.info("quadratic interpolation method search, final result {}: {}".format(res, f.subs(x, res)) )
    return res

if __name__ == '__main__':
    log.info("Unconstrained nonlinear solvers with known analysis formula")
    x = symbols('x')

    """local"""
    f = x**4 - x**2 - 2*x + 5
    newton_search(f, -5)
    quadratic_interpolation(f, -5, 6)

    """# advance and retreat
                vlb, vub = advance_retreat(f, 0, 0.1)
                golden_selection_search(f, vlb, vub, 1e-3)
                import time
                start_time = time.time()
                fibonacci_search(f, vlb, vub, 1e-3)
                log.debug("Fibonacci search, elapsed time {}".format(time.time() - start_time))
                start_time = time.time()
                fibonacci_search_reuse(f, vlb, vub, 1e-3)
                log.debug("Fibonacci search r euse, elapsed time {}".format(time.time() - start_time))"""

    """global"""
    f = x**3 - 3*x + 2
    # newton_search(f, -5) # failed to converge case, start at left of a maxima
    # newton_search_global(f, -5) # failed to converge case, start at left of a maxima
    # newton_search_global(f, -0.9)
    # vlb, vub = advance_retreat(f, -10, 0.1)
    secant_method(f, -5, 6)
    quadratic_interpolation(f, -5, 6)
    quadratic_interpolation(f, 0.5, 6)
