"""
-*- coding: utf-8 -*-
Created: ouxiaogu, 2018-07-05 10:12:38

Example code for python Build-in functions

Last Modified by: ouxiaogu
"""

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

    temperature = property(get_temperature,set_temperature)
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


if __name__ == '__main__':

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

    '''test 4, property'''
    print("\ntest 4, property:\n")
    test_Celsius()