# -*- coding: utf-8 -*-
# tested on python 3
import unittest
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

import sys
import os.path
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../")
from CnnSemJob import *

class TestCNNJob(unittest.TestCase):
    def setUp(self):
        self.myjob = MxpJob(r'../samplejob')

    def test_construct_job(self):
        myjob = Job(r'../samplejob')
        self.assertTrue(myjob.checkJobXml())

    def test_mxpjob_build(self):
        self.assertTrue(self.myjob.mxproot is not None)
        print(self.myjob.mxpCfgMap)

    @unittest.skip("only enable init")
    def test_mxpjob_enable_range_Disabled(self):
        np.testing.assert_equal(np.array([1, 2000]), np.array(self.myjob.getEnableRangeList()))

    def test_mxpjob_all_stages(self):
        stagenames, enables = zip(*self.myjob.getAllMxpStages(enabled=False))
        np.testing.assert_equal(np.array(["init", 'DLSEMCalibration']),  np.array(list(stagenames)))
        np.testing.assert_equal(np.array([1800, 2000]),  np.array(list(enables)))

    def test_mxpjob_all_stageIOfiles(self):
        print(self.myjob.getAllStageIOFiles())

    def test_mxpjob_stageIOfile(self):
        outxml = 'mxp_input.xml'
        self.assertEqual(outxml, self.myjob.getStageIOFile(enable=1800)[-len(outxml):])

    def test_dfToXmlStream(self):
        mylist = [{'name': 1, 'usage': 'CAL', 'cost_wt': 1, 'imgpath': '1_se.bmp'},
                {'name': 2, 'usage': 'VER', 'cost_wt': 1, 'imgpath': '1_se.bmp'}]
        df = pd.DataFrame(mylist)
        print(dfToXmlStream(df))

    @unittest.skip("run")
    def test_run(self):
        self.myjob.run()

def test_config():
    nodestr = """<pattern>
    <kpi>0.741096070383657</kpi>
    <test>
        <key>name</key>
        <value>213.</value>
        <options><enable>1-2000</enable></options>
    </test>
    <name>13</name>
    </pattern>"""
    root = ET.fromstring(nodestr)
    kpi = root.find(".kpi")
    print(kpi.tag, len(kpi))
    test = root.find(".test")
    print (test.tag, len(test))
    print (root.find("./test/options/enable").text)

    print (root.tag)
    print (getConfigMap(root))
    print (getUniqKeyConfigMap(root))

if __name__ == "__main__":
    # unittest.main()

    test_config()
