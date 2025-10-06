import unittest
import statsmodels
import patsy as pt
import pandas as pd
import numpy as np
import sklearn
import statsmodels.discrete.discrete_model as dm
from sklearn.ensemble import RandomForestClassifier


# Import your code from parent directory
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from assignment2 import model, modelFit, pred

# Run the checks

class testCases(unittest.TestCase):
    def testFittedModel(self):
        if hasattr(modelFit, 'tree_'):
            self.assertTrue(True)
        elif hasattr(modelFit, 'coef_'):
            self.assertTrue(True)
        elif isinstance(modelFit, RandomForestClassifier):
            self.assertTrue(True)
        elif hasattr(model, '_Booster'):
            self.assertTrue(True)
        else:
            self.assertTrue(False)
            
