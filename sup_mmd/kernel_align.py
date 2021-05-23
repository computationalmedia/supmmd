from sup_mmd.data import MMD_Dataset
import numpy as np
import torch
import sys
from itertools import product
import scipy as sp
from functions import nz_median_dist

np.set_printoptions(precision=3)
lambdaa = 0.1
dataset_name = sys.argv[1].lower()
set_ = sys.argv[2]
perfect = sys.argv[3]
assert dataset_name in {'duc03', 'duc04', 'tac08', 'tac09'}
if dataset_name in {'tac08', 'tac09'}:
	assert set_ in {'A', 'B'}
else:
	assert set_ == 'A'
assert perfect in {"y", "R.hm"}
target_name = "y_hm_0.4"
# target_name = "y_R2_0.0"
data = MMD_Dataset.load( "{}_{}".format(dataset_name, target_name), "./data/", compress = False )
def standard_scaler(x):
    return (x - np.mean(x, axis = 0)) / ( np.std(x, axis = 0) )

def describe(K):
	return np.quantile(K[np.nonzero(K)], [0.01, 0.25, 0.5, 0.75, 0.99])

def align1(K, y):
	nums = K.shape[2]
	A = np.zeros(nums)
	for m in range(nums):
		Km = K[:, :, m].squeeze()
		N = Km.shape[0]
		one = np.ones(N)
		sm = Km.sum(axis = 0)

		Kc = Km  - 1./ N * np.outer(one, sm) - 1./ N * np.outer(sm, one) + \
				1./ (N ** 2) * Km.sum() * np.outer(one, one)

		denom = len(y) * np.sqrt( ( Kc * Kc).sum() )
		numer = ( Kc * np.outer(y, y) ).sum()
		A[m] = numer/denom
		# print(numer, denom)
	return A / sum(A)

def align_helper(K, y):
	assert K.shape[0] == len(y)
	nums = K.shape[2]
	M = np.zeros((nums, nums))
	Kc = np.zeros_like(K)
	a = np.zeros(nums)
	for m in range(nums):
		Km = K[:, :, m].squeeze()
		N = Km.shape[0]
		one = np.ones(N)
		sm = Km.sum(axis = 0)

		Kc[:, :, m] = Km  - 1./ N * np.outer(one, sm) - 1./ N * np.outer(sm, one) + \
						1./ (N ** 2) * Km.sum() * np.outer(one, one)

		a[m] = ( Kc[:, :, m].squeeze() * np.outer(y, y) ).sum()
		assert np.abs(Kc.sum()) < 1e-8, Kc.sum()
	
	for m, n in product(range(nums), range(nums)):
		if m > n: continue
		M[m, n] = (Kc[:, :, m] * Kc[:, :, n]).squeeze().sum()
		M[n, m] = M[m, n]
	
	print(M, a)
	return M, a

def align_linear(M, a):
	mu1 = np.linalg.solve(M, a)
	mu1 = mu1 / np.linalg.norm(mu1)
	return mu1

def align_convex(M, a):
	v = sp.optimize.nnls(sp.linalg.sqrtm(M), sp.linalg.inv(sp.linalg.sqrtm(M)).dot(a) )[0]
	mu = v / np.linalg.norm(v)
	mu = mu / mu.sum()
	return mu

def f(v, M, a):
	obj, deriv = 0.0, 0.0
	for g in range(len(a)):
		obj += M[g].dot(v).dot(v) - 2.0 * v.dot(a[g])
		deriv += ( M[g].dot(v) - a[g] )
	return 0.5*obj/len(a), deriv/len(a)

def align_convex_combined(Ms, As):
	num_K = As[0].shape[0]
	opt = sp.optimize.minimize(f,  
			np.ones(num_K),
	        args = (Ms, As), 
	        method='L-BFGS-B', jac = True,
	        bounds = list( zip(np.zeros(num_K), np.ones(num_K)) ),
	        options={'maxiter': 100, 'disp': False, 
	        		'ftol': 1e-6, 'gtol': 1e-5 },
	        # callback = lambda x: print(x)
    )
	# print(opt)
	wts = opt["x"] / np.linalg.norm(opt["x"])
	return wts / sum(wts)

Ms, As = [], []
vlin, vconv, vconv2, valign = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

def mmd_f(gamma, Ks, Ss):
	gamma = gamma[0]
	obj = 0
	grad = 0
	for K, S in zip(Ks, Ss):
		D = K - 1.0
		obj += np.exp(gamma * D).mean()
		obj -= 2 * np.exp(gamma * D[:, S] ).mean()
		obj += np.exp(gamma * D[:, S][S ,:]).mean()

		grad += ( D * np.exp(gamma*D)).mean()
		grad -= 2*(D[:, S] * np.exp(gamma*D[:, S])).mean()
		grad += ( D[:, S][S ,:] * np.exp(gamma * D[:, S][S ,:])).mean()

	return -obj/len(Ks), -np.array([grad])/len(Ks)

def opt_gamma(K, S):
	opt = sp.optimize.minimize(mmd_f,  
			np.array([1.0]),
	        args = (K, S), 
	        method='L-BFGS-B', jac = True,
	        bounds = [(0.001, 100.0)],
	        options={'maxiter': 100, 'disp': False, 'ftol': 1e-8, 'gtol': 1e-6 },
	        callback = lambda x: print(x, -mmd_f(x, K, S)[0])
    )
	# print(opt)
	gamma = opt["x"]
	return gamma


def mmd_f_comp(gamma, KAs, KBs, KABs, Ss, lambdaa):
	gamma1 = gamma[0]
	gamma2 = gamma[0]
	obj, obj2 = 0.0, 0.0
	grad, grad2 = 0.0, 0.0
	for KA, KB, KAB, S in zip(KAs, KBs, KABs, Ss):
		DA = KA - 1.0
		DB = KB - 1.0
		DAB = KAB - 1.0
		
		obj += np.exp(gamma1 * DB).mean()
		obj -= 2 * np.exp(gamma1 * DB[:, S] ).mean()
		obj += np.exp(gamma1 * DB[:, S][S ,:]).mean()

		obj2 += np.exp(gamma2 * DA).mean()
		obj2 -= 2 * np.exp(gamma2 * DAB[:, S] ).mean()
		obj2 += np.exp(gamma2 * DB[:, S][S ,:]).mean()

		grad += ( DB * np.exp(gamma1*DB)).mean()
		grad -= 2*(DB[:, S] * np.exp(gamma1*DB[:, S])).mean()
		grad += ( DB[:, S][S ,:] * np.exp(gamma1 * DB[:, S][S ,:])).mean()

		grad2 += ( DA * np.exp(gamma2*DA)).mean()
		grad2 -= 2*(DAB[:, S] * np.exp(gamma2*DAB[:, S])).mean()
		grad2 += ( DB[:, S][S ,:] * np.exp(gamma2 * DB[:, S][S ,:])).mean()

	return (-obj + lambdaa * obj2)/len(Ss), -np.array([grad -lambdaa * grad2, 0.0])/len(Ss)

def opt_gamma_comp(KA, KB, KAB, S, lambdaa):
	
	opt = sp.optimize.minimize(mmd_f_comp,  
			np.array([1.0, 1.0]),
	        args = (KA, KB, KAB, S, lambdaa), 
	        method='L-BFGS-B', jac = True,
	        bounds = [(0.001, 100.0), (0.001, 100.0)],
	        options={'maxiter': 100, 'disp': False, 'ftol': 1e-8, 'gtol': 1e-6 },
	        callback = lambda x: print(x, -mmd_f_comp(x, KA, KB, KAB, S, lambdaa)[0])
    )
	# print(opt)
	gamma = opt["x"]
	return gamma

KAs, KBs, KABs, Ss = [], [], [], []
for ix in range(len(data)):
	group = data.groups[ix]
	K = data.kernels[(group, set_ )].numpy()

	V = data.group_idxs[(group, set_ )].numpy()
	S = data.summ_idxs[(group, set_ )].numpy()
	if perfect == "y":
		y = -1 * np.ones_like(V)
		y[S] = 1
	elif perfect == "R.hm":
		R2 = data.df['R2.P'].iloc[V].values
		R1 = data.df['R1.P'].iloc[V].values
		y = 2 * R1 * R2 / (R1 + R2 + 1e-9)
		y = standard_scaler(y)
	assert K.shape[0] == K.shape[1] == V.shape[0] == y.shape[0], \
		( K.shape[0], K.shape[1], V.shape[0], y.shape[0] )
	print("group ", group)
	
	M, a = align_helper(K, y)
	Aconv = align_convex( M, a )
	# Alin = align_linear( M, a )
	Aconv2 = align_convex_combined([M], [a])
	Aalign = align1(K, y)

	print("w, ", Aalign, Aconv, Aconv2)
	# dconv = describe(np.einsum('ijk,k->ij', K, Aconv))
	# dlin = describe(np.einsum('ijk,k->ij', K, Alin))
	dconv2 = describe(np.einsum('ijk,k->ij', K, Aconv2))
	dalign = describe(np.einsum('ijk,k->ij', K, Aalign))
	dbigrams = describe(np.einsum('ijk,k->ij', K, np.array([0.0, 1.0, 0.0])))
	print("summ, ", dalign, dconv2, dbigrams)

	if dataset_name in {"tac08", "tac09"}:
		KA, KB, KAB, _, _, SA, SB, _, _ = data[ix]
		KA, KB, KAB = KA.squeeze(), KB.squeeze(), KAB.squeeze()
		if set_ == "A":
			KA = np.einsum('ijk,k->ij', KA, Aconv2)
			print("medK A:{:.3g}".format( describe(KA)[2] ))
			KAs.append(KA)
			Ss.append(SA)
			# gamma_opt = opt_gamma([KA], [SA])
		else:
			KA = np.einsum('ijk,k->ij', KA, Aconv2)
			KB = np.einsum('ijk,k->ij', KB, Aconv2)
			KAB = np.einsum('ijk,k->ij', KAB, Aconv2)
			print("medK A:{:.3g}, B:{:.3g}, AB:{:.3g}".format(
				describe(KA)[2], describe(KB)[2], describe(KAB)[2])
			)
			KAs.append(KA)
			KBs.append(KB)
			KABs.append(KAB)
			Ss.append(SB)
			# gamma_opt = opt_gamma_comp([KA], [KB], [KAB],[SB], lambdaa)
	else:
		KA, _, SA, _ = data[ix]
		KA = KA.squeeze()
		KA = np.einsum('ijk,k->ij', KA, Aconv2)
		print("medK A:{:.3g}".format( describe(KA)[2] ))
		KAs.append(KA)
		Ss.append(SA)
		# gamma_opt = opt_gamma([KA], [SA])
	# print('gamma ', gamma_opt)
	print()

	Ms.append(M)
	As.append(a)
	# vlin += Alin
	vconv += Aconv
	vconv2 += Aconv2
	valign += Aalign

print("align:", valign / len(data))
# print("lin:", vlin / len(data))
print("conv:", vconv / len(data))
print("conv2:", vconv2 / len(data))

print("conv comb, ", align_convex_combined(Ms, As) )

if set_ == 'A':
	print("gamma ", opt_gamma(KAs, Ss ))
else:
	print("gamma ", opt_gamma_comp(KAs, KBs, KABs, Ss, lambdaa ))