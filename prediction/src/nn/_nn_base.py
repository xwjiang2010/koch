import time
from abc import ABCMeta, abstractmethod

from prediction.src.ts_base import TsPredictor


class BaseMLP(TsPredictor, metaclass=ABCMeta):
    def score(self, X, y, sample_weights):
        """
        Return the mean accuracy on the given test data and labels.
        """
        ## TODO
        return 0
