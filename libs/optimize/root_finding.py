# coding: utf-8
import numpy as np
from sympy import *
import sys
sys.path.insert(0, r'C:\Users\peyang\Perforce\peyang_LT324319_3720\app\mxp\scripts\util')
import logger
logger.initlogging(debug=True)
log = logger.getLogger("Root finding")

"""
Root finding problem is to solve x let f(x) = 0,
https://en.wikipedia.org/wiki/Root-finding_algorithm
Root-finding algorithms
- Bracketing (no derivative): Bisection method
- Quasi-Newton
    - False position
    - Secant method
- Newton
    - Newton's method
- Hybrid methods
    - Brent's method
- Polynomial methods
    - Bairstow's method
    -Jenkins–Traub method
"""

def secant_method(f, x0, x1, eps=1e-6):
    """
    secant method(切割法) use the information of target function, with previous two steps to the position guess for current step.

    General secant method is built on f(x) to find the root for f(x)=0, then the secant search
    guess is:

        x_{n+1} = x_{n} - f(x_{n})/[( f(x_{n}) - f(x_{n-1}))/(x_{n} - x_{n-1})]

    compared with the newton search guess:

        x_{n+1} = x_{n} - f(x_{n})/f'(x_{n})

    We can find that, it's just an approximation for f''(x)

        f'(x) ~= (f(x_{n}) - f(x_{n-1}))/(x_{n} - x_{n-1})

    Limitation:
    Just find the nearest root to the start point

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
        epsilon, terminate at abs( f(x) ) < epsilon

    Returns
    -------
    res : float
        the optimal value found by solver
    """
    x = list(f.free_symbols)[0]

    maxIteration = 100
    iterator = 0
    solverFailed = False
    res = x0

    f_x0 = f.subs(x, x0)
    f_x1 = f.subs(x, x1)
    if abs(f_x0) < eps:
        log.debug("secant method search, converged, check f_x0 -> {}".format(f_x0))
        res = x0
    elif abs(f_x1) < eps:
        log.debug("secant method search, converged, check f_x1 -> {}".format(f_x1))
        res = x1
    else:
        while iterator < maxIteration:
            f_x0 = f.subs(x, x0)
            f_x1 = f.subs(x, x1)

            log.debug("secant method search, iteration {}, x0 {}, f(x0) {}; x1 {}, f(x1) {}".format(iterator, x0, f_x0, x1, f_x1))

            if abs(f_x1) < eps:
                log.debug("secant method search, converged, check f_x1 -> {}".format(f_x1))
                break
            if abs(f_x1 - f_x0) == 0 :
                log.info("secant method search, failed to converge: check f_x0 {} f_x1 {} ".format(f_x0, f_x1))
                solverFailed = True
                break
            x2 = x1 - 1.0*(x1 - x0)*f_x1/(f_x1 - f_x0)

            x0 = x1
            x1 = x2
            iterator += 1

        res = x1
    log.info("secant method search, final result {}: {}".format(res, f.subs(x, res)) )
    return res

if __name__ == '__main__':
    log.info("root finding solvers with known analysis formula")
    x = symbols('x')

    f = x**2 - 2 # to solve sqrt(2)
    secant_method(f, 0, 5)
