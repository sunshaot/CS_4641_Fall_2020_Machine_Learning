import numpy as np
import pandas as pd
import statsmodels.api as sm


class FeatureReduction(object):
    def __init__(self):
        pass

    @staticmethod
    def forward_selection(data, target, significance_level=0.05):
        raise NotImplementedError

    @staticmethod
    def backward_elimination(data, target, significance_level = 0.05):
        raise NotImplementedError
