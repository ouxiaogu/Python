import unittest
import sys
import os.path
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../")
from CnnSemJob import *
import numpy as np

class TestCNNJob(unittest.TestCase):
    def setUp(self):
        self.myjob = MXPJob(r'../data/samplejob')

    def test_construct_job(self):
        myjob = Job(r'../data/samplejob')
        self.assertTrue(myjob.checkJobXml())

    def test_mxpjob_build(self):
        self.assertTrue(self.myjob.mxproot is not None)
        print(self.myjob.mxpCfgMap)

    def test_mxpjob_enable_range(self):
        np.testing.assert_equal(np.array([1, 2000]), np.array(self.myjob.getEnableRange()))

    def test_mxpjob_all_stages(self):
        stagenames, enables = zip(*self.myjob.getAllMxpStages())
        np.testing.assert_equal(np.array(["init", 'DLSEMCalibration']),  np.array(list(stagenames)))
        np.testing.assert_equal(np.array([1800, 2000]),  np.array(list(enables)))

    def test_mxpjob_all_stageIOfiles(self):
        print(self.myjob.getAllStageIOFiles())

    def test_mxpjob_stageIOfile(self):
        self.assertEqual('dlsemcal2000out.xml', self.myjob.getStageIOFile(enable=2000))

if __name__ == "__main__":
    unittest.main()
