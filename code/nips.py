import pathlib

import polars as pl
import numpy as np

""" module to read datasets from Chipman (2010) """

class Nips:
    """
    Represents one of the datasets of Chipman et al. (2010).
    sourcedir = path to single dataset directory
    split = number in [0, ..., nplits - 1]
    fold = number in [1, ..., nfolds], a negative fold indicates all folds
           except one
    """
    
    def __init__(self, sourcedir):
        source = pathlib.Path(sourcedir)
        info = pl.read_csv(source / 'info.txt', separator=' ')[:-1]
        x = pl.read_csv(source / 'x.txt', separator=' ', has_header=False, infer_schema_length=None)
        assert len(x.columns) == len(info)
        categoricals = [
            f'column_{col}'
            for col, dtype in zip(info['column.'], info['var_type'])
            if dtype == 'c'
        ]
        self._p = len(x.columns)
        self._x = x.to_dummies(columns=categoricals)
        self._y = np.loadtxt(source / 'y.txt')
        self._partitions = np.loadtxt(source / 'partitions.txt', dtype=int)
    
    @property
    def nsplits(self):
        return self._partitions.shape[1]
    
    @property
    def nfolds(self):
        return np.max(self._partitions)

    @property
    def n(self):
        return len(self._y)

    @property
    def p(self):
        return self._p
    
    def _partition(self, split):
        return self._partitions[:, split]
    
    def _traincond(self, split, fold):
        part = self._partition(split)
        if fold is None:
            return part > 0
        elif fold < 0:
            return (part > 0) & (part != -fold)
        elif fold > 0:
            return part == fold
        else:
            raise ValueError('fold == 0')
        
    def xtrain(self, *, split=0, fold=None):
        return self._x.filter(self._traincond(split, fold))
    
    def xtest(self, *, split=0):
        part = self._partition(split)
        return self._x.filter(part == 0)

    def ytrain(self, *, split=0, fold=None):
        return self._y[self._traincond(split, fold)]
    
    def ytest(self, *, split=0):
        part = self._partition(split)
        return self._y[part == 0]
