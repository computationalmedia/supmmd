from submodular.functions import *
from submodular.feature_based_functions import *
from submodular.base import *

class Mmd2Comp(Function):
    '''
        F(S+s) - F(S) for MMD^2 comparative
        is submodular
    '''

    def __init__(self, W, y, g, lambdaa, diff = 'diff', **kwargs):
        assert diff in {'diff', 'div'}
        super().__init__(len(W))
        self._W = W
        
        assert len(self._W) == len(y)
        self._lambdaa = lambdaa
        assert self._lambdaa >= 0.0 and self._lambdaa <= 1.0
        self._d = 1.0 if diff == 'div' else (1 - self._lambdaa)
        f = self.f_hard if len(set(y)) == 2 else self.f_soft
        self._Pg, self._fg = f(y, g)
        self._r = np.mean(W * self._fg, axis = 1) ##
        assert np.all(self._r > -1e8)
        self._cache_cov = 0.0
        self._cache_div = 0.0
        self._cache_vec = np.zeros(self._N)
    
    def add2S(self, s):
        self._cache_cov += self._r[s] ## set first time
        self._cache_div += 2.0 * np.dot(self._W[s, :], self._S) ## set first time
        self._cache_div += self._W[s, s]
        self._cache_vec += self._W[:, s]
        return super().add2S(s)

    def f_(self, pgx, pg):
        return pgx/pg - self._lambdaa * ( 1-pgx) / (1.0- pg)

    def f_soft(self, y, g, **kwargs ):
        pgx = y*g/2.0 + 0.5
        pg = np.sum(pgx)/len(y)
        return pg, self.f_(pgx, pg)
    
    def f_hard(self, y, g, **kwargs):
        pgx = np.array(y == g, dtype = np.int)
        assert np.sum(pgx) > 0, "check if you passed hard labels"
        pg = np.sum(pgx)/len(y)
        return pg, self.f_(pgx, pg)	
    
    def F(self, S = None, **kwargs):
        lenS = (self._nS if S is None else np.sum(S)) 
        cov = self._cache_cov if S is None else np.dot(self._r, S)
        div = 0.0
        if lenS > 0:
            div = ( self._cache_div if S is None else np.dot(np.dot(self._W, S), S))
        return 1. / lenS * ( 2.0 * cov - self._d * div / lenS)

    def marginal_gain(self, S = None, **kwargs):
        lenS = (self._nS if S is None else np.sum(S)) 
        cov = self._r.copy()
        if lenS > 0:
            cov -=  ( self._cache_cov if S is None else np.dot(self._r, S) ) / lenS
        div = - 1.0 * self._W.diagonal()
        if lenS > 0:
            div -= 2 * (self._cache_vec if S is None else np.dot(S, self._W))
            div += (2.0 * lenS + 1.0) / (lenS ** 2) * ( self._cache_div if S is None else np.dot(np.dot(self._W, S), S))
        return 1.0 / (lenS + 1.0) * (2.0 * cov + self._d * div / (lenS + 1.0))

# class KLComp(Function):
#     """
#         KL Divergence bases
#         D: bag of semantic units
#     """
#     def __init__(self, D, **kwargs):
#         super().__init__(len(D))
#         self._D = D
#         self._D_counter = Counter(chain(*D))
#         self._D_size = sum(self._D_counter.values())

#         self._S_counter = Counter()
#         self._S_size = 0

#     def add2S(self, s):
#         self._S_counter += Counter(self._D[s])
#         self._S_size += len(self._D[s])
#         return super().add2S(s)
    
#     def F(self, S = None, **kwargs):
#         if S is not None:
#             S_counter = Counter(self._D[np.where( S == 1)[0]])
#             S_size = sum(S_counter.values())
#             return kl_counters(S_counter, self._D_counter, S_size, self._D_size)
#         return kl_counters(self._S_counter, self._D_counter, self._S_size, self._D_size)
    
#     def marginal_gain(self, S = None, **kwargs):
#         msk = ((self._S if S is None else S) == 1)
#         res = np.zeros(self._N)
#         if S is not None:
#             S_counter = Counter(self._D[np.where(msk)[0]])
#             S_size = sum(S_counter.values()) 
#         else:
#             S_counter = self._S_counter
#             S_size = self._S_size
#         F_S = kl_counters(S_counter, self._D_counter, S_size, self._D_size)
#         for s in np.where(~msk)[0]:
#             F_Ss = kl_counters(
#                 S_counter + Counter(self._D[s]),
#                 self._D_counter,
#                 S_size + len(self._D[s]),
#                 self._D_size
#             )
#             res[ix] = (F_Ss - F_S)
#         return res

class Li3Comp(Function):
    def __init__(self, W, y, g, lambdaa1 = 1.0, lambdaa2 = 1.0, **kwargs):
        super().__init__(len(W))
        assert W.shape == (self._N, self._N)
        self._W = W
        assert len(y) == self._N
        self._lambdas = [lambdaa1, lambdaa2]
        self._g = g
        self._y = y
        self._fg = (self._y == self._g)*np.ones(self._N)-self._lambdas[1]*(y != g)*np.ones(self._N)
        self._r = (W * self._fg).sum(axis = 1)
        assert np.all(self._r > - 1e8), self._r[np.where(self._r < - 1e8)[0]]
        self._cache_sum = 0.0
        self._cache_vec = np.zeros(self._N)
        self._cache = 0.0

    def add2S(self, s):
        self._cache += self._r[s] ## set first time
        self._cache_sum += 2.0 * np.dot(self._W[s, :], self._S) ## set first time
        self._cache_sum += self._W[s, s]
        self._cache_vec += self._W[:, s]
        return super().add2S(s)

    def F(self, S = None, **kwargs):
        cov = self._cache if S is None else np.dot(self._r, S)
        div = self._cache_sum if S is None else np.dot(np.dot(self._W, S), S)
        return cov - self._lambdas[0] * div

    def marginal_gain(self, S = None, **kwargs):
        res = -1e6 * np.ones(self._N)
        idxs = np.where(np.logical_and((self._S if S is None else S) == 0, self._y == self._g))[0]
        # idxs = np.where((self._S if S is None else S) == 0)[0]
        cov = self._r[idxs].copy()
        div = self._W.diagonal()[idxs]
        div += 2 * (self._cache_vec[idxs] if S is None else np.dot(S, self._W[:, idxs]))
        res[idxs] = cov - self._lambdas[0] * div
        return res
    
    def curvature(self, **kwargs):
        c = np.zeros(self._N, dtype = np.float32)
        V_s = np.ones(self._N, dtype = np.bool) * (self._y == self._g)
        idxs = np.where(self._y == self._g )[0]
        for s in idxs:
            V_s[s] = False
            val = 1.0 * self.marginal_gain(S = V_s)[0] / self( np.logical_not(V_s ))
            c[s] = val
            V_s[s] = True
        return 1 - np.min(c)
