"""
-*- coding: utf-8 -*-
Created: peyang, 2018-01-24 15:17:55

Last Modified by:  ouxiaogu

XmlUtil: Tachyon Job Xml handling module
"""

from xml.etree import ElementTree
import xml.etree.ElementTree as ET
import re
import pandas as pd
import logger
from StrUtil import parseText
from collections import deque

log = logger.setup("XmlUtil")

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

    def setChildNodeVal(self, pattern, intext=''):
        p = self.root.find(pattern)
        child = p.find('value')
        child.text = intext

    def getChildNodeVal(self, pattern):
        try:
            p = self.root.find(pattern)
            child = p.find('value')
            return child.text
        except:
            log.error("Cannot find the value of pattern %s in root" % pattern)
            return ''

def addChildNode(node, key='', val=None):
    child = ET.Element(key)
    if val is not None:
        child.text = val
    node.append(child)
    return child

def getChildNode(node, pattern, count=0, suppress_warn=False):
    if not pattern.startswith("."): # depth=1
        pattern = "."+pattern
        log.debug("add '.' before pattern to only search depth=1: %s", pattern)
    if '/' in pattern:
        log.warning("Warning, please avoid ambiguous by removing '/' in pattern")
    for i, inode in enumerate(node.findall(pattern)):
        if i==count:
            return inode
    if suppress_warn:
        log.warning("Warning, getChildNode, Failed to find No. {} child with pattern {} in giving node".format(count, pattern))
    return None

def setChildNodeVal(node, pattern, intext='', count=0):
    p = getChildNode(node, pattern, count)
    if p is not None:
        p.text = intext

def getChildNodeVal(node, pattern, count=0, defaultval=None):
    p = getChildNode(node, pattern, count)
    if p is not None:
        return getElemText(p)
    return defaultval

def getElemText(elem_, trim=False, precision=3):
    """
    Parse text of xml Elem

    Parameters
    ----------
    elem_:  xml Elem object,
        if elem_ node has children, raise ValueError;
        else return parsed text
    """
    text_ = elem_.text
    if text_ is None:
        return ""
    else:
        return parseText(text_, trim, precision)

def getSonConfigMap(elem, trim=False, precision=3):
    """
    Parse the tag & text of all depth=1 children node into a map.

    Example
    -------
    Return {'kpi': 0.741, 'name': 13}
    """
    rst = {}
    for item in list(elem):
        if len(item) > 0:
            continue
        rst[item.tag] = getElemText(item, trim, precision)
    return rst

def getUniqKeyConfigMap(elem):
    """
    Recursively parse the tag & text of all children nodes into a map.

    Example
    -------
    Return {'kpi': 0.741096070383657, 'name': 13, 'test': {'value': 211.0}}
    """
    rst = {}
    for item in list(elem):
        if len(item) > 0:
            rst[item.tag] = getUniqKeyConfigMap(item)
        else:
            rst[item.tag] = getElemText(item)
    return rst

def getConfigMapList(node, pattern):
    """
    For all the items met pattern type under current node,
    based on the result of getConfigMapBFS, output a list of KWs map
    suitable for Xml Tree, with multiple pattern node, but every pattern node
    have unique children.
    """
    rst = []
    if not pattern.startswith("."): # depth=1
        pattern = "."+pattern
        log.debug("add '.' before pattern to only search depth=1: %s", pattern)
    for child in node.findall(pattern):
        curMap = getFullConfigMap(child)
        rst.append(curMap)
    return rst

def getFullConfigMap(elem):
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
    for inode in list(elem): # inode, just get tag
        curtag = './'+inode.tag
        if curtag not in visited:
            for j, jnode in enumerate(elem.findall(curtag)): # jnode, parse all children matched pattern
                visited.append(curtag)
                nodekey = inode.tag if j==0 else inode.tag+'@'+str(j)
                if len(jnode) > 0:
                    jrst = getFullConfigMap(jnode)
                    jrst = {(nodekey+'/'+k): v for k, v in jrst.items()}
                else:
                    jrst = {nodekey: getElemText(jnode)}
                rst = {k: v for d in (rst, jrst) for k, v in d.items()}
    return rst

def getConfigMapBFS(elem):
    '''
    Examples
    --------
    Return {'kpi': 0.741096070383657, 'name': 13, 'test/key': 'name', 'test/value': 213.0, 'test/value@1': 212.0, 'test@1/value': 211.0, 'test/options/enable': '1-2000'}
    '''
    tags = deque([(elem, [''], [0])])
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

def getConfigMapDFS(elem):
    '''
    Examples
    --------
    Return {'kpi': 0.741096070383657, 'name': 13, 'test/key': 'name', 'test/value': 213.0, 'test/value@1': 212.0, 'test/options/enable': '1-2000', 'test@1/value': 211.0}
    '''
    tags = deque([(elem, [''], [0])])
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

def dfFromConfigMapList(node, pattern):
    return pd.DataFrame(getConfigMapList(node, pattern))

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
        nextnode = getChildNode(curnode, pattern=tag, count=idx, suppress_warn=True)
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

def dfToMxpXml(df):
    root = ET.Element('root')
    ocf = ET.Element('result')
    root.append(ocf)
    for _, series in df.iterrows():
        occf = dfRowToXmlNode(series)
        ocf.append(occf)
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
    print ('getChildNodeVal(root, ".test/value")', getChildNodeVal(root, ".test/value"))
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
    root = dfToMxpXml(df)
    indent(root)
    tree = ET.ElementTree(root)
    tree.write("example.xml", encoding="utf-8", xml_declaration=True)

    root = ET.parse('example.xml').getroot().find('.result')
    print ('\ngetChildNodeVal(root, "/value")', getChildNodeVal(root, ".pattern/value"))
    df = dfFromConfigMapList(root, '.pattern')
    print(df)
