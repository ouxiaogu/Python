# -*- coding: utf-8 -*-
"""
Created: peyang, 2019-01-17 11:59:03

Decision Tree class:
1. various cost functions
2. various optimization algos

Last Modified by:  ouxiaogu
"""
import numpy as np
from Classifier import Classifier

class GenericTree(object):
    """
    Generic tree node

    Example
    -------
    #    *
    #   /|\
    #  1 2 +
    #     / \
    #    3   4
    t = Tree('*', [Tree('1'),
                   Tree('2'),
                   Tree('+', [Tree('3'),
                              Tree('4')])])
    """
    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

    def set_children(self, nodes):
        for node in nodes:
            self.add_child(node) 

    def set_class(self, label):
        self.label = label

class BinTree(object):
    """binTree: binary tree"""
    def __init__(self, name='root', left=None, right=None):
        self.name = name
        self.left = None
        self.right = None
        self.label = None

    def set_class(self, label):
        self.label = label

class ClfTree(Classifier):
    """
    ClfTree class: Decision Tree for Classification

    Parameters
    ----------
    criterion:  str
        DT split criterion, could be 'gini', 'gain', 'gain_ratio', case insensitive
    eps:    float
        the threshold to accept a splitting or not, for gain, it's > eps; for gini, it's < eps
    """
    def __init__(self, criterion='gini', eps=None, name='root'):
        super(ClfTree, self).__init__()
        self.criterion = criterion.lower()
        if eps is None:
            eps = 0.5 if self.criterion == 'gini' else 1e-9
        self.eps = eps

    def fit(X, y):
        pass

    @classmethod
    def build_tree(cls, X, y):
        return cls.build_tree(X, y)

    @staticmethod
    def value_counts(y_):
        '''
        Count how many instances in each label among all the target value (class label)
        return dict counts
        '''
        y = y_
        if instances(y_, pd.Series):
            y = y_.values()

        seen = set()
        return {v: len(list(filter(lambda tmp: tmp==v, y))) for v in y if not (v in seen or seen.add(v))}

    @staticmethod
    def entropy(y):
        '''
        No matter the attribute is continuous or discrete, the target value (class label) is discrete for 
        Classification ClfTree. So the entropy function is common for all:

        $H(X) &= H(p) = -\sum_{i=1}^{n} p_i \log p_i$
        '''
        counts = ClfTree.value_counts(y)
        return -sum([p*np.log2(p) for p in np.array(list(counts.values()), dtype='float')/len(y)])

    @staticmethod
    def gini_impurity(y):
        '''gini coefficient/impurity '''
        counts = ClfTree.value_counts(y)
        return 1 - sum([p^2 for p in np.array(list(counts.values()), dtype='float')/len(y)])

    @classmethod
    def info_gain(cls, y, X, A, thres=None):
        '''
        3 cost types: info_gain, info_gain_ratio & gini, all use classmethod, like factory pattern
        Able to change the exact conditional cost function under specific class type
        '''
        return self.entropy(y) - cls.cond_entropy(y, X, A, thres)

    @classmethod
    def info_gain_ratio(cls, y, X, A, thres=None):
        return (self.entropy(y) - cls.cond_entropy(y, X, A, thres)) / self.entropy(X.ix[:, A])

    @classmethod
    def gini(cls, y, X=None, A=None, thres=None):
        '''
        For entropy & info gain, we usually use max{info gain} to greedily choose subtree
        For Gini impurity, we just use min{conditional Gini} to greedily choose subtree
        So here add this API for usage 
        '''
        if (x is None) or (A is None):
            return ClfTree.gini_impurity(y)
        else:
            return cls.cond_gini(y, X, A, thres)

    def parse_criterion(self):
        split_criterion_func = None
        valid_split_gain = None
        if self.criterion == 'info_gain':
            split_criterion_func = self.info_gain
            valid_split_gain = lambda g : g > self.eps
        elif self.criterion == 'info_gain_ratio':
            split_criterion_func = self.info_gain_ratio
            valid_split_gain = lambda g : g > self.eps
        elif self.criterion == 'gini':
            split_criterion_func = self.gini
            valid_split_gain = lambda g : g < self.eps
        else:
            sys.exit("split criterion should be info_gain, info_gain_ratio or gini\n")
        return split_criterion_func, valid_split_gain

class DiscreteTree(ClfTree):
    """
    DiscreteTree for DT with discrete attribute values
    its internal node is `A=?`, 
        - Binary tree: left subtree A==?, right subtree A!=?
        - Generic Tree: subtree T_i root at A=a (*include binary tree, implement GenericTree for DiscreteTree)
    Add an unused variable thres=None under DiscreteTree, for keeping an uniform factory API at the parent ClfTree class. 
    """
    def __init__(self, *arg, **kwargs):
        super(DiscreteTree, self).__init__(*arg, **kwargs)

    @staticmethod
    def cond_entropy(y, X, A, thres=None):
        '''
        implemented based on pandas

        $H(Y|X)&=\sum_{i=1}^n p_i H(Y|X=x_i)  & p_i=P(X=x_i),i=1,2,\dots ,n$

        Parameters
        ----------
        y: pandas Series 
            target value(class label)
        X: pandas DataFrame 
            N instances, each with n features
        A: str
            Attribute/column name, which feature to be split on
        Returns
        -------
        Hda: float
            Conditional entropy
        '''
        labels = set(X.ix[:, A].values)
        return sum(map(lambda sr: len(sr)*ClfTree.entropy(sr), [y.ix[X.ix[:, A] == k] for k in labels])) / len(y)

    @staticmethod
    def cond_gini(y, X, A, thres=None):
        '''Conditional Gini coefficient, @see DiscreteTree.cond_entropy'''
        labels = set(X.ix[:, A].values)
        return sum(map(lambda sr: len(sr)*ClfTree.gini_impurity(sr), [y.ix[X.ix[:, A] == k] for k in labels])) / len(y)

    def build_tree(X, y, name='root'):
        tree = GenericTree(name)

        #1. stop criterion
        if len(set(y)) == 1:
            tree.set_class(set(y)[0])
            return tree
        #2. stop criterion 2
        if len(X.columns) == 0:
            counts = self.value_counts(y)
            max_label = max(counts, key = lambda k: counts[k])
            tree.set_class(max_label)
            return tree

        #3. split tree
        split_criterion_func, valid_split_gain = self.parse_criterion()
        fea_best = max(X.columns, key = lambda fea: split_criterion_func(y, X, fea))

        #4. split benefit fail the epsilon
        if valid_split_gain(self.gini(y, X, fea_best)):
            counts = self.value_counts(y)
            max_label = max(counts, key = lambda k: counts[k])
            tree.set_class(max_label)
            return tree
        #5. split into sub dataset, subtrees, recursively
        else:
            fea_values = sorted(set(X.ix[:, fea_best]))
            subtrees = [ build_tree(X.ix[X[fea_best]==v, filter(lambda t: t!=fea_best, X.columns)], 
                                    y.ix[X[fea_best]==v], v)  for v in fea_values]
            tree.add_children(subtrees)
        return tree

class ContinuousTree(ClfTree):
    """
    ContinuousTree for DT with continuous attribute values
    Implement BinTree for ContinuousTree, only support tree pruning in ContinuousTree
    its internal node is `A>=a`, left subtree A<=a, right subtree A>a
    """
    def __init__(self, *arg, **kwargs):
        super(ContinuousTree, self).__init__(*arg, **kwargs)
        self.tree = BinTree(self.name)

    @staticmethod
    def cond_entropy(y, X, A, thres):
        '''@see DiscreteTree.cond_entropy'''
        labels = set(X.ix[:, A].values)
        y_split = [y.ix[X.ix[:, A] <= thres], y.ix[X.ix[:, A] > thres]]
        return sum(map(lambda sr: len(sr)*ClfTree.entropy(sr), y_split)) / len(y)

    @staticmethod
    def cond_gini(y, X, A, thres):
        '''Conditional Gini coefficient, @see DiscreteTree.cond_entropy'''
        labels = set(X.ix[:, A].values)
        y_split = [y.ix[X.ix[:, A] <= thres], y.ix[X.ix[:, A] > thres]]
        return sum(map(lambda sr: len(sr)*ClfTree.gini_impurity(sr), y_split)) / len(y)

    @staticmethod
    def get_split_feature_thres(y, X):
        '''
        based on C4.5 improvement on the continuous attributes, we choose info gain as criterion to get the best split feature and split location
        '''
        return fea_best, thres_best 

    def build_tree(self, X, y, name='root'):
        tree = BinTree(name)

        #1. stop criterion
        if len(set(y)) == 1:
            tree.set_class(set(y)[0])
            return tree
        #2. stop criterion 2
        if len(X.columns) == 0:
            counts = self.value_counts(y)
            max_label = max(counts, key = lambda k: counts[k])
            tree.set_class(max_label)
            return tree

        #3. split tree
        split_criterion_func, valid_split_gain = self.parse_criterion()
        fea_best, thres_best = self.get_split_feature_thres(y, X)

        #4. split benefit fail the epsilon
        if valid_split_gain(split_criterion_func(y, X, fea_best, thres_best)):
            counts = self.value_counts(y)
            max_label = max(counts, key = lambda k: counts[k])
            tree.set_class(max_label)
            return tree
        #5. split into sub dataset, subtrees, recursively
        else:
            fea_values = sorted(set(X.ix[:, fea_best]))
            subtrees = [self.build_tree(X.ix[X[fea_best]==v, filter(lambda t: t!=fea_best, X.columns)], 
                                       y.ix[X[fea_best]==v], v)  for v in fea_values]
            tree.add_children(subtrees)
        return tree

    def prune(self):
        pass
        
if __name__ == '__main__':
    a = 2*[1] + 3*[2]
    print(ClfTree.value_counts(a))
    print(ClfTree.Entropy(a))
