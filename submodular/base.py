# from _future_ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy, copy
import types
import numpy as np
from profilehooks import profile

class Function(ABC):
    '''
        A class for computing mar ginal gain (discrete derivative) over candidates
            N: Number of items in set, indexed as 1..N

            other notations used inside:
            S: indicator for selected set cached
    '''

    def __init__(self, N):
        self._N = N ## size of ground set (V) indexed as 1..N
        self._S = np.zeros(self._N, dtype = bool) ## state of summary
        self._nS = 0
    
    def __len__(self):
        return self._nS
        
    @property
    def universe_size(self):
        return self._N

    def add2S(self, s):
        assert type(int(s)) == int, s
        self._nS += 1
        self._S[s] = True
    
    def __repr__(self):
        return "%s(%d | %d)"%(self.__class__.__name__, self._S.sum(), self._N)
    
    @property
    def S(self):
        return self._S
    
    @abstractmethod
    def F(self, S = None, **kwargs):
        '''
            for debugging (empirically checking submodularity)
            F(S)
        '''
        raise NotImplementedError("Must override function")

    # @profile
    def marginal_gain(self, S = None, **kwargs):
        '''
            returns: 
                F(S+s) - F(S) for each s (aka discrete derivative \\del_s F(S))
        '''
        
        res = -1e9 * np.ones(self._N, dtype = np.float)
        FS = self(self._S if S is None else S)
        S1 = deepcopy(self._S if S is None else S)

        for ix in range(self._N):
            if S1[ix]: continue
            S1[ix] = True
            res[ix] = self(S1) - FS
            S1[ix] = False
        return res

    def __call__(self, *args, **kwargs):
        return self.F(*args, **kwargs)
    
    def curvature(self, **kwargs):
        c = np.zeros(self._N, dtype = np.float32)
        V_s = np.ones(self._N, dtype = bool)
        for s in range(self._N):    
            V_s[s] = False
            val = 1.0 * self.marginal_gain(S = V_s)[s] / self(1 - V_s)
            c[s] = val
            V_s[s] = True
        return 1 - np.min(c)

class FeatureBasedF(Function):
    '''
        F(S+s) - F(S) for feature based submodular function
        form: F(S) = \sum_i w_i \phi(\sum_s x_is), think X as item-feature interaction matrix (bipartite graph)
            - indices i over features, s over data points; phi is concave function
            - w_i is feature weights (e.g. idf scores for bag-of-words)
            - features can be similarity vector to all data points, (e.g. coverage function in Lin Bilmes 11)
            - or cluster assignment (one hot encoded) multiplied by reward of data point (e.g. diversity in Lin Bilmes 11)
    '''

    def __init__(self, X, w = None, **kwargs):
        super().__init__(X.shape[0])
        self._cacheU = np.zeros(X.shape[1])
        if w is None:
            w = np.ones(X.shape[1])
        self._w = w
        assert len(self._w) == len(self._cacheU), (len(self._w), len(self._cacheU))
        self._X = X # or override
        self.concaveF = lambda x: x

    def add2S(self, s):
        self._cacheU += self._X[s, :]
        return super().add2S(s)
    
    def F(self, S = None, **kwargs):
        ## mean to normalize, doesn't matter if sum is used
        lenS = np.sum(self._S if S is None else S)
        temp = self._cacheU if S is None else np.dot(self._X.T, S)
        return np.dot(self._w, self.concaveF(temp))
    
    def marginal_gain(self, S = None, **kwargs):
        # s = np.where( (self._S if S is None else S) == 0)[0]
        lenS = np.sum(self._S if S is None else S)
        temp = self._cacheU if S is None else np.dot(self._X.T, S)
        gain = self.concaveF(temp + self._X) - self.concaveF(temp)
        return np.dot(gain, self._w)
