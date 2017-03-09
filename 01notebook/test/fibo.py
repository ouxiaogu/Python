# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 16:57:29 2017

@author: peyang
"""

def fib(n):    # write Fibonacci series up to n
    a, b = 0, 1
    while b < n:
        print b,
        a, b = b, a+b
if __name__ == "__main__":
    print dir(__name__)
    fib(400)