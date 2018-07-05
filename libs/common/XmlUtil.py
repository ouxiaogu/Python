"""
-*- coding: utf-8 -*-
Created: peyang, 2018-01-24 15:17:55

Last Modified by: ouxiaogu

XmlUtil: Tachyon Job Xml handling module
"""

import xml.etree.ElementTree as ET
import re
import pandas as pd
import logger
from StrUtil import parseText

logger.initlogging(debug=False)
log = logger.getLogger("XmlUtil")

def addChildNode(node, key='', val=''):
    """
    Add key-value type child node in Brion job xml way

    Args:
        node: parent xml node to put the child

    kwargs:
        key: key    -> 'name' tag
        val: value  -> 'value' tag

    Example:
        >>> addChildNode(node, "jobtype", "mxp")
        <item>
          <name>jobtype</name>
          <value>mxp</value>
        </item>
    """
    child = ET.Element('item')
    grandchild = ET.Element('name')
    grandchild.text = key
    child.append(grandchild)
    grandchild = ET.Element('value')
    grandchild.text = val
    child.append(grandchild)
    node.append(child)
    return node

def setChildNodeVal(node, pattern, intext=''):
    p = node.find(pattern)
    child = p.find('value')
    child.text = intext
    return node

def getChildNodeVal(node, pattern):
    try:
        p = node.find(pattern)
        child = p.find('value')
        return child.text
    except:
        raise KeyError("Cannot find the value of pattern %s in node" % pattern)

def getElemText(elem_, trim=False, precision=3):
    """
    Parse text of xml Elem

    Args:
        elem_:  xml Elem object,
                if elem_ node has children, raise ValueError;
                else return parsed text
    """
    text_ = elem_.text
    if text_ is None:
        return ""
    else:
        return parseText(text_, trim, precision)

def getConfigMap(elem, trim=False, precision=3):
    """
    Parse the tag & text of all depth=1 children node into a map.

    Example:
        <pattern>
            <kpi>0.741096070383657</kpi>
            <name>13</name>
        </pattern>
        >>> getConfigMap(elem)
        {'kpi': 0.741, 'name': 13}
    """
    rst = {}
    for item in list(elem):
        if len(item) > 0:
            continue
        rst[item.tag] = getElemText(item)
    return rst

def getRecurConfigMap(elem):
    """
    Recursively parse the tag & text of all children nodes into a map.
    """
    rst = {}
    for item in list(elem):
        if len(item) > 0:
            rst[item.tag] = getRecurConfigMap(item)
        else:
            rst[item.tag] = getElemText(item)
    return rst

def getConfigMapList(node, pattern):
    """
    For all the items met pattern type under current node,
    based on the result of getConfigMap, output a list of KWs map"""
    rst = []
    if not pattern.startswith("."): # depth=1
        pattern = "."+pattern
        log.debug("add '.' before pattern to only search depth=1: %s", pattern)
    for child in node.findall(pattern):
        curMap = getConfigMap(child)
        rst.append(curMap)
    return rst

def dfFromConfigMapList(node, pattern):
    return pd.DataFrame(getConfigMapList(node, pattern))

if __name__ == '__main__':
    nodestr = """<pattern>
    <kpi>0.741096070383657</kpi>
    <test>
        <key>name</key>
        <value>213.</value>
    </test>
    <name>13</name>
</pattern>"""
    root = ET.fromstring(nodestr)
    kpi = root.find(".kpi")
    test = root.find(".test")
    print (getChildNodeVal(root, ".test"))
    # print getChildNodeVal(root, ".test2")
    print (len(test))
    print (len(kpi))

    print (root.tag)
    print (getConfigMap(root, trim=True))
    print (getRecurConfigMap(root))