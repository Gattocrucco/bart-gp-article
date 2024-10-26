from rpy2 import robjects

from . import _base

class bart(_base.RObjectABC):

   _rfuncname = 'BayesTree::bart'

   def __init__(self, *args, seed=None, **kw):
        if seed is not None:
            robjects.r(f'set.seed({seed})')
        super().__init__(*args, **kw)
