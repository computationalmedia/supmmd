from submodular.base import FeatureBasedF, Function
import numpy as np
from profilehooks import profile
from sklearn.preprocessing import OneHotEncoder

class LBDiv(FeatureBasedF):
	'''
		F(S+s) - F(S) from Lin and Bilmes 2011 paper's diversity function
		is submodular
	'''

	def __init__(self, W, partitions, **kwargs):
		enc = OneHotEncoder(categories='auto')
		self._r = W.sum(axis = 0 )
		assert W.shape[0] == len(self._r)
		assert len(partitions) == len(self._r), ( len(partitions), len(self._r) )
		X = enc.fit_transform( np.array([partitions]).T ).toarray()
		X1 = np.multiply(X.T, self._r).T
		super(LBDiv, self).__init__(X = X1, w = 1./np.sqrt(W.shape[0]) * np.ones(X1.shape[1], dtype = np.float ))
		self.concaveF = np.sqrt

class LBCov(FeatureBasedF):
	'''
		F(S+s) - F(S) from Lin and Bilmes 2011 paper's coverage function
		is submodular
	'''

	def __init__(self, W, alpha = 1.0, **kwargs):
		super(LBCov, self).__init__(X = W)
		self._alpha = alpha
		self._r = self._alpha * W.sum(axis = 0 )
		assert self._alpha >= 0.0 and self._alpha <= 1.0
		assert W.shape[0] == len(self._r)
		self.concaveF = lambda x: np.minimum(x , self._r)

class LinBilmes(Function):
	'''
		F(S+s) - F(S) from Lin and Bilmes 2011 paper's function
		is submodular
	'''

	def __init__(self, W, partitions, alpha = 0.5, lambdaa = 2.0, **kwargs):
		self._cov = LBCov(W, alpha)
		self._div = LBDiv(W, partitions)
		self._lambda = lambdaa
		assert self._lambda >= 0.0
		super().__init__(W.shape[0])
	
	def add2S(self, s):
		self._cov.add2S(s)
		self._div.add2S(s)
		return super().add2S(s)

	def F(self, S = None, **kwargs):
		if S is None:
			assert np.all(self._cov._S == self._div._S)
			assert np.all(self._S == self._div._S)
		return self._cov.F(S) + self._lambda * self._div.F(S)
	
	# @profile
	def marginal_gain(self, S = None, **kwargs):
		if S is None:
			assert np.all(self._cov._S == self._div._S)
			assert np.all(self._S == self._div._S)
		return self._cov.marginal_gain(S) + \
			self._lambda * self._div.marginal_gain(S)