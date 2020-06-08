import unittest
import numpy as np
from lits.lits_dataset import LitsDataSet
from lits.helpers import plot_patient_samples

d = LitsDataSet('')
d.x = np.ones((5, 100, 100))
d.y = np.ones((5, 100, 100))
d.slices = [[0, 4]]

class LitsHelperTests(unittest.TestCase):

    def test_plot_patient_samples(self):
        plot_patient_samples([0], d)
        
    def test_plot_patient_samples_other_set(self):
        plot_patient_samples([0], d, set="y")