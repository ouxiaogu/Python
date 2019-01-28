# -*- coding: utf-8 -*-
"""
Created: peyang, 2019-01-17 11:59:03

Decision Tree class:
1. various cost functions
2. various optimization algos

Last Modified by:  ouxiaogu
"""
import sys
import numpy as np
import pandas as pd
from Classifier import Classifier
from sklearn.model_selection import train_test_split

class Tree(object):
    """docstring for Tree"""
    def __init__(self):
        self.name = None
        self._label = None
        self.counts = {}
        self.split_criterion = {}
        pass

    @property
    def label(self):
        """I'm the 'x' property."""
        return self._label

    @label.setter
    def label(self, rhs):
        self._label = rhs

    def prunedErrors(self):
        '''number of errors if pruning current tree into leaf node'''
        return sum([v for k, v in self.counts if self.label != k])

class GenericTree(Tree):
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
        super(GenericTree, self).__init__()
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, node):
        assert(isinstance(node, Tree))
        self.children.append(node)

    def add_children(self, nodes):
        for node in nodes:
            self.add_child(node) 

    def describe(self):
        print('name： {}, counts: {}, label: {}, criterion: {}'.format(self.name, str(self.counts), self.label, str(self.split_criterion)))
        for child in self.children:
            if child is not None:
                child.describe()

class BinTree(Tree):
    """binTree: binary tree"""
    def __init__(self, name='root', left=None, right=None):
        super(BinTree, self).__init__()
        self.name = name
        self.parent = None
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None
    
    def set_left(self, subtree):
        assert(isinstance(subtree, Tree))
        subtree.parent = self.name
        self.left = subtree

    def set_right(self, subtree):
        assert(isinstance(subtree, Tree))
        subtree.parent = self.name
        self.right = subtree

    def size(self):
        sz = 0
        if self.is_leaf():
            return 1
        else:
            if self.left is not None:
                sz += self.left.size()
            if self.right is not None:
                sz += self.right.size()
        return sz

    def numOfErrors(self):
        num = 0
        if self.is_leaf():
            return self.prunedErrors()
        else:
            if self.left is not None:
                num += self.left.numOfErrors()
            if self.right is not None:
                num += self.right.numOfErrors()
        return num

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

    @staticmethod
    def value_counts(y_):
        '''
        Count how many instances in each label among all the target value (class label)
        return dict counts
        '''
        y = y_
        if isinstance(y_, pd.Series):
            y = y_.values

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
        return 1 - sum([p**2 for p in np.array(list(counts.values()), dtype='float')/len(y)])

    @classmethod
    def info_gain(cls, y, X=None, A=None, thres=None):
        '''
        3 cost types: info_gain, info_gain_ratio & gini, all use classmethod, like factory pattern
        Able to change the exact conditional cost function under specific class type
        '''
        if (X is None) or (A is None):
            return 0
        else:
            return cls.entropy(y) - cls.cond_entropy(y, X, A, thres)

    @classmethod
    def info_gain_ratio(cls, y, X=None, A=None, thres=None):
        if (X is None) or (A is None):
            return 0
        else:
            return (cls.entropy(y) - cls.cond_entropy(y, X, A, thres)) / cls.entropy(X.ix[:, A])

    @classmethod
    def gini(cls, y, X=None, A=None, thres=None):
        '''
        For entropy & info gain, we usually use max{info gain} to greedily choose subtree
        For Gini impurity, we just use min{conditional Gini} to greedily choose subtree
        So here add this API for usage 
        '''
        if (X is None) or (A is None):
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
    def __init__(self, **kwargs):
        super(DiscreteTree, self).__init__(**kwargs)

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

    def fit(self, X, y):
        return self.build_tree(X, y)
    
    def build_tree(self, X, y):
        tree = GenericTree()
        counts = self.value_counts(y)
        max_label = max(counts, key = lambda k: counts[k])
        tree.label = max_label
        tree.counts = counts

        #1. stop criterion
        if len(set(y)) == 1:
            return tree
        #2. stop criterion 2
        if len(X.columns) == 0:
            return tree

        #3. split tree
        split_criterion_func, valid_split_gain = self.parse_criterion()
        fea_best = max(X.columns, key = lambda fea: split_criterion_func(y, X, fea))

        #4. split benefit fail the epsilon
        split_benifit = split_criterion_func(y, X, fea_best)
        if not valid_split_gain(split_benifit):
            return tree
        #5. split into sub dataset, subtrees, recursively
        else:
            tree.name = fea_best
            tree.split_criterion = {self.criterion: split_benifit}
            fea_values = sorted(set(X.ix[:, fea_best]))
            subtrees = [ self.build_tree(X.ix[X[fea_best]==v, filter(lambda t: t!=fea_best, X.columns)], 
                                    y.ix[X[fea_best]==v])  for v in fea_values]
            tree.add_children(subtrees)
        return tree

class ContinuousTree(ClfTree):
    """
    ContinuousTree for DT with continuous attribute values
    Implement BinTree for ContinuousTree, only support tree pruning in ContinuousTree
    its internal node is `A>=a`, left subtree A<=a, right subtree A>a
    """
    def __init__(self, **kwargs):
        super(ContinuousTree, self).__init__(**kwargs)
        self.tree = BinTree(self.name)

    @staticmethod
    def cond_entropy(y, X, A, thres):
        '''@see DiscreteTree.cond_entropy'''
        y_split = [y.ix[X.ix[:, A] <= thres], y.ix[X.ix[:, A] > thres]]
        return sum(map(lambda sr: len(sr)*ClfTree.entropy(sr), y_split)) / len(y)

    @staticmethod
    def cond_gini(y, X, A, thres):
        '''Conditional Gini coefficient, @see DiscreteTree.cond_entropy'''
        y_split = [y.ix[X.ix[:, A] <= thres], y.ix[X.ix[:, A] > thres]]
        return sum(map(lambda sr: len(sr)*ClfTree.gini_impurity(sr), y_split)) / len(y)

    def split_feature_and_thres(self, y, X):
        '''
        based on C4.5 improvement on the continuous attributes, we choose info gain as criterion to get the best split feature and split location
        '''
        split_eps = 0.05 # hard code for info_gain_ratio & gini_impurity, a valid split need info_gain > split_eps
        if self.criterion == 'info_gain':
            split_eps = self.eps
        
        fea_best = X.columns[0]
        thres_best= None
        gain_best= 0
        memo = {}
        for fea in X.columns:
            fea_vals = sorted(set(X.ix[:, fea].values))
            cur_thres_best = None if len(fea_vals) == 1 else np.mean(fea_vals[:2])
            cur_gain_best = 0
            for i in range(len(fea_vals) - 1):
                cur_thres = np.mean(fea_vals[i:(i+2)])
                cur_gain = self.info_gain(y, X, fea, cur_thres)
                if cur_gain >= split_eps: # only valid if met criterion
                    if cur_gain_best > cur_gain:
                        cur_gain_best = cur_gain
                        cur_thres_best = cur_thres
            memo[fea] = (cur_gain_best, cur_thres_best)
            if gain_best > cur_gain_best:
                gain_best = cur_gain_best
                thres_best = cur_thres_best
        return fea_best, thres_best

    def fit(self, X, y):
        return self.build_tree(X, y)

    def build_tree(self, X, y):
        tree = BinTree()

        counts = self.value_counts(y)
        max_label = max(counts, key = lambda k: counts[k])
        tree.label = max_label

        #1. stop criterion
        if len(set(y)) == 1:
            return tree
        #2. stop criterion 2
        if len(X.columns) == 0:
            return tree

        #3. split tree
        split_criterion_func, valid_split_gain = self.parse_criterion()
        fea_best, thres_best = self.split_feature_and_thres(y, X)

        #4. split benefit fail the epsilon
        if not valid_split_gain(split_criterion_func(y, X, fea_best, thres_best)):
            return tree
        #5. split into sub dataset, subtrees, recursively
        else:
            tree.name = '{} <= {:.3f}'.format(fea_best, thres_best)
            tree.set_left(self.build_tree(X.ix[X[fea_best] <= thres_best, filter(lambda t: t!=fea_best, X.columns)],
                                        y.ix[X[fea_best] <= thres_best]))
            tree.set_right(self.build_tree(X.ix[X[fea_best] > thres_best, filter(lambda t: t!=fea_best, X.columns)],
                                        y.ix[X[fea_best] > thres_best]))
        return tree

    def predict(self, X, tree):
        y_pred = []
        for _, x in X.iterrows():
            y_pred.append(self.predict_single(x, tree) )
        return pd.Series(y_pred, index=X.index)

    @staticmethod
    def predict_single(x, tree):
        if tree.is_leaf():

            return tree.label
        else:
            if len(x.query(tree.name)) > 0 :
                return ContinuousTree.predict_single(x.left)
            else:
                return ContinuousTree.predict_single(x.right)

    def prune(self, tree, X_test, y_test):
        '''
        Implemented here is, use cross-validation method, training set to build the decision tree T, 
        test set to prune T and get pruned tree P.
        '''
        treeSize = tree.size()
        self.feed_test_set(tree, X_test, y_test )
        prunedTreeSize, prunedTree = ContinuousTree.pruning(tree)
        print("Tree #T: {}, after pruning, #T = {}".format(treeSize, prunedTreeSize))
        return prunedTree
    
    @staticmethod
    def pruning(tree):
        if tree is None:
            return None

        numOfErrs = tree.numOfErrors()
        if tree.is_leaf():
            return 1, tree

        if tree.prunedErrors() <= numOfErrs:
            tree.left = tree.right = None
            prunedTree = tree
            return 1, prunedTree
        else:
            tree.left = ContinuousTree.pruning(tree.left)
            tree.right = ContinuousTree.pruning(tree.right)
            treeSize = tree.size()
        return treeSize, tree

    def feed_test_set(self, tree, X, y_true):
        tree.counts = ClfTree.value_counts(y_true)
        if len(y_true) == 0 or tree.is_leaf():
            return
        flt = X.eval(tree.name)
        if tree.left is not None:
            self.feed_test_set(tree.left, X.ix[flt, :], y_true.ix[flt])
        if tree.right is not None:
            self.feed_test_set(tree.right, X.ix[~flt, :], y_true.ix[~flt])
        return

if __name__ == '__main__':
    df = pd.read_csv("./Input/mdata_5-1.txt", index_col=0)
    cols = df.columns.tolist()
    X = df[cols[:-1]]
    y = df[cols[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    '''
    import sys
    from io import TextIOWrapper
    sys.stdout = TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    '''

    print(df.columns.values, X_test, y_test, sep='\n')
    
    clf = DiscreteTree(eps=0.02, criterion="info_gain")
    tree = clf.fit(X_train, y_train)
    print()
    tree.describe()