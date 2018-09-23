"""
-*- coding: utf-8 -*-
Created: peyang, 2018-01-24 15:17:55

Last Modified by: ouxiaogu

XmlUtil: Xml handling module

* Tree level: ElementTree is  
    - Tree level: Those naming are all for node level

    `Elem`, `cf`, `gcf`, `root`

"""

import xml.etree.ElementTree as ET
import re
import pandas as pd
import logger
from StrUtil import parseText
from collections import deque

log = logger.setup("XmlUtil")

__all__ = ['JobInfoXml', 'addChildNode', 'getChildNode', 
            'setConfigData', 'getConfigData', 
            'getSonConfigMap', 'getUniqKeyConfigMap', 'getFullConfigMap', 
            'dfFromConfigMapList', 'dfToMxpOcf', ]

class JobInfoXml(object):
    """
    only for the xml files in Tachyon job jobinfo

    <item>
      <name>jobtype</name>
      <value>mxp</value>
    </item>
    """
    def __init__(self, root):
        super(JobInfoXml, self).__init__()
        self.root = root

    def addChildNode(self, key='', val=''):
        """
        Add key-value type child node in Brion job xml way

        Returns
        -------
            key: key    -> 'name' tag
            val: value  -> 'value' tag

        Example
        -------
            >>> addChildNode("jobtype", "mxp")
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
        self.root.append(child)

    def setConfigData(self, key, intext=''):
        p = self.root.find(key)
        child = p.find('value')
        child.text = intext

    def getConfigData(self, key):
        try:
            p = self.root.find(key)
            child = p.find('value')
            return child.text
        except:
            log.error("Cannot find the value of key %s in root" % key)
            return ''

def addChildNode(cf, key='', val=None):
    child = ET.Element(key)
    if val is not None:
        child.text = val
    cf.append(child)
    return child

def getChildNode(cf, key, count=0, suppress_warn=False):
    if not key.startswith("."): # depth=1
        key = "."+key
        log.debug("add '.' before key to only search depth=1: %s", key)
    if '/' in key:
        log.warning("Warning, please avoid ambiguous by removing '/' in key")
    for i, inode in enumerate(cf.findall(key)):
        if i==count:
            return inode
    if suppress_warn:
        log.warning("Warning, getChildNode, Failed to find No. {} child with key {} in giving cf".format(count, key))
    return None

def setConfigData(cf, key, val='', count=0):
    p = getChildNode(cf, key, count)
    if p is not None:
        p.text = val
    else:
        addChildNode(cf, key, val)

def getConfigData(cf, key, defaultval=None, count=0):
    p = getChildNode(cf, key, count)
    if p is not None:
        return getElemText(p)
    return defaultval

def getGlobalConfigData(gcf, cf, key, defaultval=None):
    lval = getConfigData(cf, key, defaultval)
    if lval is None or lval == defaultval:
        return getConfigData(gcf, key, defaultval)
    return lval

def getElemText(elem, trim=False, precision=3):
    """
    Parse text of xml Elem

    Parameters
    ----------
    elem:  xml Elem object,
        if elem node has children, raise ValueError;
        else return parsed text
    """
    text = elem.text
    if text is None:
        return ""
    else:
        return parseText(text, trim, precision)

def getSonConfigMap(cf, trim=False, precision=3):
    """
    Parse the tag & text of all depth=1 children node into a map.

    Example
    -------
    Return {'kpi': 0.741, 'name': 13}
    """
    rst = {}
    for item in list(cf):
        if len(item) > 0:
            continue
        rst[item.tag] = getElemText(item, trim, precision)
    return rst

def getUniqKeyConfigMap(cf):
    """
    Recursively parse the tag & text of all children nodes into a map.

    Example
    -------
    Return {'kpi': 0.741096070383657, 'name': 13, 'test': {'value': 211.0}}
    """
    rst = {}
    for item in list(cf):
        if len(item) > 0:
            rst[item.tag] = getUniqKeyConfigMap(item)
        else:
            rst[item.tag] = getElemText(item)
    return rst

def getConfigMapList(cf, key):
    """
    For all the items met key type under current cf,
    based on the result of getConfigMapBFS, output a list of KWs map
    suitable for Xml Tree, with multiple key node instances
    """
    rst = []
    if not key.startswith("."): # depth=1
        key = "."+key
        log.debug("add '.' before key to only search depth=1: %s", key)
    for child in cf.findall(key):
        curMap = getFullConfigMap(child)
        rst.append(curMap)
    return rst

def getFullConfigMap(cf):
    '''
    parse the complete xml tree into a flat depth=2 KWs map

    Examples
    --------
    Return {'kpi': 0.741096070383657, 'name': 13
    'test/key': 'name', 'test/value': 213.0, 'test/value@1': 212.0, 'test/options/enable': '1-2000', 
    'test@1/value': 211.0}
    '''
    rst = {}
    visited= []
    for inode in list(cf): # inode, just get tag
        curtag = './'+inode.tag
        if curtag not in visited:
            for j, jnode in enumerate(cf.findall(curtag)): # jnode, parse all children matched key
                visited.append(curtag)
                nodekey = inode.tag if j==0 else inode.tag+'@'+str(j)
                if len(jnode) > 0:
                    jrst = getFullConfigMap(jnode)
                    jrst = {(nodekey+'/'+k): v for k, v in jrst.items()}
                else:
                    jrst = {nodekey: getElemText(jnode)}
                rst = {k: v for d in (rst, jrst) for k, v in d.items()}
    return rst

def getConfigMapBFS(cf):
    '''
    Examples
    --------
    Return {'kpi': 0.741096070383657, 'name': 13, 'test/key': 'name', 'test/value': 213.0, 'test/value@1': 212.0, 'test@1/value': 211.0, 'test/options/enable': '1-2000'}
    '''
    tags = deque([(cf, [''], [0])])
    rst = {}
    while len(tags) > 0:
        visited = []
        curnode, abspaths, indices = tags.popleft()
        for i, inode in enumerate(list(curnode)):
            reltag = './'+inode.tag
            new_abspaths = abspaths.copy()
            new_abspaths.append(reltag)
            if reltag not in visited:
                for j, jnode in enumerate(curnode.findall(reltag)):
                    new_indices = indices.copy()
                    new_indices.append(j)
                    postfixes = ['' if ix==0 else '@'+str(ix) for ix in new_indices]
                    nodekey = ''.join([a+b for a, b in zip(new_abspaths, postfixes)])
                    if len(jnode) > 0:
                        tags.append((jnode, new_abspaths, new_indices))
                    else:
                        rst[nodekey] = getElemText(jnode)
                visited.append(reltag)
        # print(visited)
    final = {(re.sub(r'\./', r'/', k)[1:]): v for k, v in rst.items()}
    return final

def getConfigMapDFS(cf):
    '''
    Examples
    --------
    Return {'kpi': 0.741096070383657, 'name': 13, 'test/key': 'name', 'test/value': 213.0, 'test/value@1': 212.0, 'test/options/enable': '1-2000', 'test@1/value': 211.0}
    '''
    tags = deque([(cf, [''], [0])])
    rst = {}
    while len(tags) > 0:
        visited = []
        curnode, abspaths, indices = tags.popleft()
        for i, inode in enumerate(list(curnode)):
            reltag = './'+inode.tag
            abstag = ''.join(abspaths)
            new_abspaths = abspaths.copy()
            new_abspaths.append(reltag)
            if reltag not in visited:
                for j, jnode in enumerate(curnode.findall(reltag)):
                    new_indices = indices.copy()
                    new_indices.append(j)
                    postfixes = ['' if ix==0 else '@'+str(ix) for ix in new_indices]
                    nodekey = ''.join([a+b for a, b in zip(new_abspaths, postfixes)])
                    if len(jnode) > 0:
                        if abstag =='':
                            tags.append((jnode, new_abspaths, new_indices))
                        else:
                            tags.insert(0, (jnode, new_abspaths, new_indices))
                    else:
                        rst[nodekey] = getElemText(jnode)
                visited.append(reltag)
        # print(visited)
    final = {(re.sub(r'\./', r'/', k)[1:]): v for k, v in rst.items()}
    return final

def dfFromConfigMapList(cf, key):
    return pd.DataFrame(getConfigMapList(cf, key))

def updateXmlPath(root, paths_indice, value):
    '''
    updateXmlPath, 
        if xml path exists, then update the value; 
        if not, then add node, and update the value
    '''
    paths_indice = deque(paths_indice)
    curnode = root
    while len(paths_indice) != 0:
        tag, idx = paths_indice.popleft()
        # print(tag, idx)
        nextnode = getChildNode(curnode, key=tag, count=idx, suppress_warn=True)
        if nextnode is None:
            if len(paths_indice) == 0:
                nextnode = addChildNode(curnode, tag, str(value))
            else:
                nextnode = addChildNode(curnode, tag)
        curnode = nextnode

def dfRowToXmlNode(rowSeries, tag='pattern', path_sep='/', index_sep='@'):
    """output DataFrame row into a xml pattern node"""
    series = rowSeries.dropna()
    node = ET.Element(tag)
    for tag in series.index:
        paths_indice = [c for c in tag.split(path_sep)]
        paths_indice = [c.split(index_sep) for c in paths_indice]
        paths_indice = [c if len(c)==2 else c+[0]  for c in paths_indice ]
        updateXmlPath(node, paths_indice, series[tag])
    return node

def dfToMxpOcf(df):
    root = ET.Element('root')
    ocf = ET.Element('result')
    root.append(ocf)
    for _, series in df.iterrows():
        occf = dfRowToXmlNode(series)
        ocf.append(occf)
    indent(root)
    return root

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

if __name__ == '__main__':
    nodestr = """<pattern>
    <kpi>0.741096070383657</kpi>
    <name>13</name>
    <test>
        <key>name</key>
        <value>213.</value>
        <value>212.</value>
        <options><enable>1-2000</enable></options>
    </test>
    <test>
        <value>211.</value>
    </test>
    <test>
        <value>210.</value>
        <key><option>revive_bug=Ticket111</option></key>
    </test>
    </pattern>"""

    root = ET.fromstring(nodestr)
    kpi = root.find(".kpi")
    test = root.find(".test")
    print ('getConfigData(root, ".test/value")', getConfigData(root, ".test/value"))
    print ('len(test) 1st', len(test), test.text)
    print ('len(kpi) 1st', len(kpi))

    print ('root.tag', root.tag)
    print ('getSonConfigMap', getSonConfigMap(root, trim=True))
    print ('getUniqKeyConfigMap', getUniqKeyConfigMap(root))
    print('getFullConfigMap', getFullConfigMap(root))
    print('getConfigMapBFS', getConfigMapBFS(root))
    print('getConfigMapDFS', getConfigMapDFS(root))

    # print(getConfigMapList(root, '.test'))
    df = dfFromConfigMapList(root, '.test')
    print(df)
    root = dfToMxpOcf(df)
    tree = ET.ElementTree(root)
    tree.write("example.xml", encoding="utf-8", xml_declaration=True)

    root = ET.parse('example.xml').getroot().find('.result')
    print ('getConfigData(.pattern/value)', getConfigData(root, ".pattern/value"))
    df = dfFromConfigMapList(root, '.pattern')
    print(df)
