"""
-*- coding: utf-8 -*-
Created: hshi & peyang, 2018-01-25 11:26:37

Last Modified by: ouxiaogu

StrUtil: String handling utility module
"""
from collections import OrderedDict
import re

def parseKW(content, outer_sep=',', inner_sep='='):
    """parser for string kwargs"""
    kw = [c.strip() for c in content.split(outer_sep)]
    kw = [[v.strip() for v in c.split(inner_sep)] for c in kw ]
    try:
        rst = OrderedDict(kw)
    except ValueError:
        raise ValueError('error occurs when converting %s to dict' % str(kw))
    return rst

def buildKW(dict_, outtersep=',', innersep='='):
    try:
        if not isinstance(dict_, dict):
            dict_ = dict(dict_)
    except TypeError:
        raise TypeError("Error occurs when converting to dict for: %s" % str(dict_))
    kws = map(lambda x: (x[0], str(x[1])), dict_.items())
    return outtersep.join(map(innersep.join, kws))

def parseText(text_, trim=False, precision=3):
    """
    Returns
    -------
    text_:  the conversion relationship
        string  =>  string
        numeric =>  int or float, decided by whether text_ contains '.'
    """
    if text_ is None:
        return ""
    try: # numeric
        try:
            if re.match(r"^[\n\r\s]+", text_) is not None:
                raise ValueError("Input text is empty")
        except TypeError:
            raise TypeError("error occurs when parse text %s" % text_)
        if '.' in text_: #float
            text_ = float(text_)
            if trim:
                precision = int(precision) if precision > 1 else 3
                text_ = round(text_, precision)
        else: # int
            text_ = int(text_)
    except ValueError: # string
        pass
    return text_

if __name__ == '__main__':
    options_ = "enable=1-2000,MXP.debug=1,avx=1"
    options_ = "enable = 1-2000 , MXP.debug=1, avx = 1 "
    options =  parseKW(options_)
    print (options)
    print (buildKW(options))
    print (buildKW({"x": 2}))
    # print buildKW([1,2,3])
    print (parseText("324.2"))
