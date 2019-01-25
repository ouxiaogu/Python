# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:02:55 2017

@author: hshi
"""
import numpy as np
import pandas as pd
import string
import re
from collections import namedtuple, OrderedDict
from scipy import stats

from logger import logger
log = logger.getLogger(__name__)

__all__ = ['Range', 'State', 'Command', 'Message', 'parseKW', 'buildKW',
           'snapFilterSeries', 'snapFilterDF', 'unique', 'value_counts']

QUOTASTR = '#QUOTA#'


class Range(object):
    def __init__(self, minv, maxv=None, step=None, fix=None):
        minv = 0 if minv is None else minv
        self.min = minv
        self.max = minv if maxv is None else maxv
        self.step = 1e-3 if (step is None) or np.isnan(step) else step
        if self.min > self.max:
            self.min, self.max = self.max, self.min
        if self.step <= 0:
            self.step = 1e-3
        self.fix = self.min if fix is None else fix

    def copy(self):
        return self.__class__(self.min, self.max, self.step, self.fix)

    @property
    def tuple(self):
        return (self.min, self.max, self.step)

    @property
    def middle(self):
        return (self.min + self.max) / 2.0

    @property
    def populate(self):
        return np.arange(self.min, self.max + 1e-6, self.step).tolist()

    @property
    def size(self):
        return len(self.populate)

    @property
    def isfixed(self):
        return abs(self.min - self.max) < 1e-6

    @property
    def pos(self):
        diff = [abs(v - self.fix) for v in self.populate]
        try:
            pos = diff.index(min(diff))
        except:
            pos = 0
        return pos

    def tofix(self):
        self.min = self.max = self.fix

    def __eq__(self, other):
        return ((self.min == other.min) &
                    (self.max == other.max) &
                    (self.step == other.step) &
                    (self.fix == other.fix))

    def __le__(self, other):
        if isinstance(other, set):
            return self.values.issubset(other)
        elif isinstance(other, self.__class__):
            return self.values.issubset(other.values)
        else:
            return False

    def __str__(self):
        return 'min={}, max={}, step={}, fix={}'.format(self.min, self.max, self.step, self.fix)


class State(object):
    ''' define all the available states in this class
    Please introduce the meaning of each state behind defination
    To add a new state, define it in this class, then add code in main loop to
    deal with the state
    '''
    WAIT = 'wait'    # wait for receiving command from socket
    EXIT = 'exit'    # exist the main loop


class Command(object):
    ''' Define all available commands here
    Define the command by string, because of easier read
    for each command, there is a script to send it
    '''
    WAIT = 'wait'  # optimize the current resist template on GUI
    EXIT = 'exit'   # exit the programe


class Message(object):
    SEP_BIG = ','
    SEP_SMALL = '='

    @classmethod
    def parse(cls, msg):
        ''' parse the message to namedtuple, so that user can acess each
        message via attribute style

        Notice: input message should NOT contain spaces between key-value pairs
        '''
        msg = msg.split(cls.SEP_BIG)
        msg = OrderedDict(map(lambda x: x.split(cls.SEP_SMALL), msg))
        Class = namedtuple('Class', msg.keys(), verbose=False)
        return Class(*msg.values())

    @classmethod
    def build(cls, **msg):
        ''' build massage from dict
        '''
        return cls.SEP_BIG.join(map(lambda x: cls.SEP_SMALL.join(x), msg.items()))


def snapFilterDF(df, values):
    ''' snap some columns in a DataFrame to specific values. The anlogu in pandas is:
    query('a==3, b==5'), instead, it will make (a, b) a vector, and find the rows with
    (a, b) closest distance to (3, 5). The rows with the same 1st smallest distance are
    all returned

    args:
        df: pd.DataFrame
        values: pd.Series
    returns:
        snappedvalue: pd.Series
        filteredDF: pd.DataFrame
        snapped: whether the value is snapped
    '''
    if values.size == 0:
        raise TypeError('input series should not be empty')
    if (df.index.size == 0) or (df.columns.size == 0):
        raise TypeError('input lookup table should not be empty')
    keys = values.index.tolist()
    keys = list(set(keys) & set(df.columns))
    if len(keys) == 0:
        raise IndexError('No common keys!')
    values = values[keys]

    gp = df.groupby(keys)
    idx = pd.DataFrame(gp.indices.keys(), columns=keys)
    minidx = (idx - values).abs().sort_values(by=keys).index[0]
    snapedvalues = idx.ix[minidx, :]
    diff = snapedvalues - values
    gap = diff.mul(diff).sum()
    snapped = True if gap > 0 else False
    if snapedvalues.size == 1:
        keys = snapedvalues.ix[0]
    else:
        keys = tuple(snapedvalues)
    return snapedvalues, gp.get_group(keys), snapped


def snapFilterSeries(series, value):
    ''' Snap a series to a fixed value
    args:
        series: pd.Series
        value: number
    returns:
        snappedvalue: float
        snappedindex: int
        snapped: whether the value is snapped
    '''
    if series.size == 0:
        return value, None, False
    diff = (series - value).abs()
    diff = diff.sort_values()
    snappedindex = diff.index[0]
    snappedvalue = series.ix[snappedindex]
    snapped = snappedvalue != value
    return snappedvalue, snappedindex, snapped


def snapListRange(inlist, inrange, ACCURACYTHRESH=0.0001):
    ''' Snap a Range's min, max step to match a list
    args:
    ----
    inseries: iterable
    inrange: Range

    Return:
    ----
    Range
    '''
    oversnapped = False
    inlist = pd.Series(inlist)
    snappedmin, _, _ = snapFilterSeries(inlist, inrange.min)
    snappedmax, _, _ = snapFilterSeries(inlist, inrange.max)
    filt = (inlist>=snappedmin) & (inlist<=snappedmax)
    temp = inlist.ix[filt]
    vmin, vmax = temp.min(), temp.max()
    vstep = (vmax - vmin) / float(temp.size - 1)

    if (np.abs(snappedmax - inrange.max) > ACCURACYTHRESH) or \
       (np.abs(snappedmin - inrange.min) > ACCURACYTHRESH):
           oversnapped = True
    return Range(vmin, vmax, vstep), oversnapped


class MyDataFrame(pd.DataFrame):
    def query(self, criteria):
        idx=self.apply(lambda x: eval(criteria.format(**dict(x)), {}, {'nan':np.nan}), axis=1)
        return self.ix[idx, :]

def checkCostFun(costfun, filtered=True, validation=True):
    ELEMENTLIST = ['rms', 'rms1d', 'rms2d', 'unwtrms', 'range', 'range1d', 'range2d', 'bw', 'total', 'mean', 'avg', 'sig3', 'anchor', 'ratio', 'ratio1d', 'ratio2d', 'wtratio', 'wtratio1d', 'wtratio2d']
    if filtered:
        ELEMENTLIST += ['{}_f'.format(x) for x in ELEMENTLIST]
    if validation:
        ELEMENTLIST += ['{}_v'.format(x) for x in ELEMENTLIST]
    class TEMP():
        def eval_(self, EXPRESSION, ELEMENTLIST):
            LOCALS = locals()
            from math import sqrt
            for ELEMENT in ELEMENTLIST:
                LOCALS[ELEMENT] = 1
            try:
                eval(EXPRESSION)
            except NameError as e:
                raise NameError('In cost function express, {}'.format(e.message))
    test = TEMP()
    test.eval_(costfun, ELEMENTLIST)


def pandasPow(series, power):
    return pd.Series(np.power(series, power), index=series.index, name=series.name)


def calPValue(baserms, reducerms, gaugenumber, basetermN):
    F12 = (reducerms**2 - baserms**2) * (gaugenumber - basetermN) / baserms**2
    p = stats.f.sf(F12, 1, gaugenumber - basetermN)
    return p

def unique(seq):
    # Fast way to get the unique values from an list
    # use the set.add return none after a succeeded addition
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def value_counts(seq):
    # Fast way to get value counts for an list
    # use the set.add return none after a succeeded addition
    seen = set()
    return {x: len(list(filter(lambda tmp: tmp==x, seq))) for x in seq if not (x in seen or seen.add(x))}