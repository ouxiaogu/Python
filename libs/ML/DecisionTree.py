# -*- coding: utf-8 -*-
"""
Created: peyang, 2019-01-17 11:59:03

Decision Tree class:
1. various cost functions
2. various optimization algos

Last Modified by:  ouxiaogu
"""

from Classifier import Classifier

class Tree(Classifier):
    """docstring for Tree"""
    def __init__(self, arg, criterion='Gini'):
        super(DTree, self).__init__()
        self.criterion = criterion
        self.criterion = criterion

    def fit(X, y):
        
        