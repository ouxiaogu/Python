"""
-*- coding: utf-8 -*-
Created: ouxiaogu, 2018-07-05 10:12:38

Example code for python Build-in functions

Last Modified by:  ouxiaogu
"""
import pandas as pd
import logging
logger = logging.getLogger(__name__)

def zip(*iterables):
    '''pick one elem from all the input iterables, pack into one elem for output, only terminate when one iterables come to its tail'''
    # zip('ABCD', 'xy') --> Ax By
    sentinel = object()
    iterators = [iter(it) for it in iterables]
    while iterators: # always true
        result = []
        for it in iterators:
            elem = next(it, sentinel)
            if elem is sentinel: # truly terminate criteria
                return
            result.append(elem)
        yield tuple(result) # prepare each pack result after 1 pick

class C:
    def __init__(self):
        self._x = None

    @property
    def x(self):
        """I'm the 'x' property."""
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @x.deleter
    def x(self):
        del self._x

def test_C():
    c = C()
    print(c.x )
    c.x = 10
    print(c.x )

class C2(object): # equivalent to class C
    def __init__(self):
        self._x = None

    def _x_get(self):
        return self._x
    x = property(_x_get, doc="I'm the 'x' property.")

    def _x_set(self, value):
        self._x = value
    x = x.setter(_x_set)

    def _x_del(self):
        del self._x
    x = x.deleter(_x_del)

class Celsius:
    '''a real example of property'''
    def __init__(self, temperature = 0):
        self._temperature = temperature

    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32

    def get_temperature(self):
        print("Getting value")
        return self._temperature

    def set_temperature(self, value):
        if value < -273:
            raise ValueError("Temperature below -273 is not possible")
        print("Setting value")
        self._temperature = value

    temperature = property(get_temperature, set_temperature)
    # or use @property, and ${var}.setter

def test_Celsius():
    c = Celsius(37)
    print(c.get_temperature() )
    c.set_temperature(10)

    # with property
    print(c.temperature )
    c.temperature=40
    print(c.temperature )
    print(c.to_fahrenheit() )

# eval and repl

class Date(object):
    def __init__(self, day=0, month=0, year=0):
        self.day = day
        self.month = month
        self.year = year

    @staticmethod
    def from_string1(date_as_string):
        day, month, year = map(int, date_as_string.split('-'))
        date1 = Date(day, month, year)
        return date1

    @classmethod # don't
    def from_string2(cls, date_as_string):
        day, month, year = map(int, date_as_string.split('-'))
        date1 = cls(day, month, year)
        return date1

    def tostr(self):
        return "{0}-{1}-{2}".format(self.month, self.day, self.year)

class DateTime(Date):
  def tostr(self):
      return "{0}-{1}-{2} - 00:00:00PM".format(self.month, self.day, self.year)

class Dog:
    kind = 'canine'         # class variable shared by all instances
    tricks = []             # mistaken use of a class variable

    def __init__(self, name):
        self.name = name    # instance variable unique to each instance

    def add_trick(self, trick):
        self.tricks.append(trick)

def test_Dog():
    d = Dog('Fido')
    e = Dog('Buddy')
    print(d.kind == 'canine')   # shared by all dogs
    print(e.kind == 'canine')   # shared by all dogs
    print(d.name)               # unique to d, 'Fido'
    print(e.name)               # unique to e, 'Buddy'

    # tricks will be unexpectedly as combine of all Dog instances

def all(iterable):
    for element in iterable:
        if not element:
            return False
    return True

def any(iterable):
    for element in iterable:
        if element:
            return True
    return False

def enumerate(sequence, start=0):
    n = start
    for elem in sequence:
        yield n, elem
        n += 1

def slice(start, stop, step=None):
    # a[slice] == a[start:stop:step]
    pass


def addlogger(cls: type):
    aname = '_{}__log'.format(cls.__name__)
    setattr(cls, aname, logging.getLogger(cls.__module__ + '.' + cls.__name__))
    return cls


@addlogger
class Foo(object):
    def foo(self):
        self.__log.info('foo called')


@addlogger
class Bar(Foo):
    def bar(self):
        self.__log.info('bar called')

def test_set():
    '''
    A set is an unordered collection with no duplicate elements. 
    Basic uses include 
        - membership testing
        - eliminating duplicate entries. 
    '''
    a = 2*[1] + 3*[2]
    print(pd.Series(a).value_counts().to_dict())
    print(dict(set((v, len(list(filter(lambda t: t==v, a)))) for v in a)) )

if __name__ == '__main__':
    """
    import argparse
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('BAR', help='BAR help')
    parser.add_argument('-f', '--foo', help='foo help')
    args = argparse.parse_args(['BAR', '--foo', 'FOO'])

    '''test 1, zip'''
    print("\ntest 1, zip different length\n")
    x, y = zip('ABCD', 'xy')
    print(x)
    print(y)

    '''test 2, zip'''
    print("\ntest 2, zip same length:\n")
    a = tuple((x, x*2) for x in range(5) )
    x, y = zip(*a)
    print(x)
    print(y)

    '''test 3, staticmethod/classmethod'''
    print("\ntest 3, staticmethod/classmethod:\n")
    date1 = Date.from_string1("05-07-2018")
    print(date1.tostr() )


    date2 = Date.from_string2("05-07-2018")
    print(date2.tostr() )
    date3 = DateTime.from_string2("05-07-2018")
    print(date3.tostr() )
    """

    '''test 4, property'''
    print("\ntest 4, property:\n")
    test_C()
    print()
    test_Celsius()

    """
    '''test 5, logging decorator
    bar = Bar()
    bar.foo()
    bar.bar()
    '''

    # >>> INFO:__main__.Foo:foo called
    # >>> INFO:__main__.Bar:bar called
    """
    
    """
    '''test 6, test set'''
    test_set()
    """