### Data Structure

A parent Node: (key, [])
A Leave Node: (key, value)

### Example:

1. dict

('test/value': 213.0, 'test/value@1': 212.0, 'test@1/value': 211.0, 'test@2/value': 210.0)

[(test, [(value, 213), (value, 212)]), (test, [(value, 211)], (test, [(value, 210)])
 
2. df

test/options/enable   test/value  test/value@1      test@2/key/option  test@2/value  
1-2000     213.0         212.0  revive_bug=Ticket111         210.0

Paths-indice, value
([(test, 0), (options, 0), (enable, 0)], 1-2000)
([(test, 0), (value, 0) ], 213.0)
([(test, 0), (value, 1) ], 212.0)
([(test, 2), (key, 0), (option, 0) ], revive_bug=Ticket111)
([(test, 2), (value, 0)], 210)


Merge list:

([(test, 0), (options, 0), (enable, 0)], 1-2000)
([(test, 0), (value, 0) ], 213.0)
([(test, 0), (value, 1) ], 212.0)
