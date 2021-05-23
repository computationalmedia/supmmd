import numpy as np
import torch 

def softmax(x):
	z = np.exp( x - np.max(x) )
	return z / sum(z)
	
def mmd_pq(K, S, f):
	pV = softmax(f)
	qS = softmax(f[S])
	o1 = np.dot(pV, np.dot( K, pV ))
	o2 = np.dot(pV, np.dot( K[:, S], qS ))
	o3 = np.dot(qS, np.dot( K[S, :][:, S], qS ))
	return o1 - 2.0 * o2 + o3 

def mmd_p(K, S, f):
	pV = softmax(f)
	o1 = np.dot(pV, np.dot( K, pV ))
	o2 = np.dot(K[S, :], pV).mean()
	o3 = K[S, :][:, S].mean()
	return o1 - 2.0 * o2 + o3

def mmd(K, S):
	return K.mean() - 2.0 * K[S, :].mean() + K[S, :][:, S].mean()

def nz_median_k(K):
	if type(K) != np.ndarray:
		K = K.numpy()
	# k_med = np.median( K )
	k_med = np.median( K[np.nonzero(K)] )
	return k_med

def nz_median_dist(K):
	if type(K) != np.ndarray:
		K = K.numpy()
	# k_med = np.median( K )
	k_med = np.median( K[np.nonzero(K)] )
	return 1.0 - k_med

def combine_kernels(K, alpha, gamma):
	K_combined = torch.einsum('ijk,k->ij', K, alpha)
	d = nz_median_dist(K_combined) ## around 0.95
	gamma = 2.0 * gamma / d
	K_combined = np.exp(- gamma) * torch.exp( gamma * K_combined )
	return K_combined
