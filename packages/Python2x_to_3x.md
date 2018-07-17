Python 3.x differ with Python 2.x
=================================

# Data structures

## iterations

In python 3.x, `range`, `list comprehensionn` and `map` are both iterator, not value now.

```python
a = list(range(10))

b = tuple(lambda x: x**2, x in a)

c = tuple(map(lambda x: x**2, a))
```

## function

1. print with brace `print()`
