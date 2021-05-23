import torch
from sup_mmd.data import MMD_Dataset
import sys, json, re, os, glob
from commons.utils import get_logger
import numpy as np
from sup_mmd.model import MMD_comp
from submodular.maximize import greedy as greedy_maximize
import pandas as pd
from copy import copy
from itertools import product
from sup_mmd.functions import softmax, nz_median_dist, combine_kernels
import multiprocessing as mp

logger = get_logger("Infer")
Rs = [ 0.0, 0.001, 0.003, 0.01]
GPU_MODE = False      
BUDGET = 120
TARGET_NAME = "y_hm_0.4" # shouldn't matter
ROOT = "./"

Gammas = [ 0.2, 0.4, 0.6, 0.8, 1.0 ]
# Gammas = [0.5, 1.5]
Lambdas = [0.1, 0.2, 0.3, 0.4]
val_dataset = sys.argv[1]
assert val_dataset in {"tac08", "tac09"}

ALPHAs = {
    "tac08": [
		[0.334, 0.408, 0.258],
		[0.01, 0.97, 0.02],
		[0.0, 1.0, 0.0],
    ]
}[val_dataset]

def infer(gamma1, lambdaa, MODE):
    if MODE == "val" :
        dataset = val_dataset
    else:
        dataset = {
            "tac08": "tac09",
            "tac09": "tac08"
        }[val_dataset]

    dataset_name = "{}_{}".format(dataset, TARGET_NAME)

    logger.info("loading data from " + dataset_name)
    data = MMD_Dataset.load(dataset_name, ROOT + "/data/", compress = False)

    idxs = np.arange( len(data) ).tolist()
    root = "{}/unsupB_{}/rouge_{}".format(ROOT, val_dataset, MODE)
    if not os.path.exists(root):
        try:
            os.makedirs(root)
        except:
            pass
    logger.info("Dataset loaded, begin inference with #topics={}".format(len(idxs)))

    def write_result(group, set_, subset, S, alpha_seq, r):
        set_ = ("-%s"%set_) if len(data.sets) > 1 else ""
        fname = "%s%s.M.100.X.%s_r%.2g"%( group.upper(), set_, "mmd_{}_g{}_a{}".format(dataset, gamma1, alpha_seq), r)
        sents = subset["sent"].iloc[S].values
        path_write = root + "/gamma_" + str(gamma1)
        if not os.path.exists(path_write):
            try:
                os.makedirs(path_write)
            except:
                pass
        with open(path_write + "/" + fname, "w") as fp:
            fp.write("\n".join(sents))

    for alpha_seq, alpha in enumerate(ALPHAs):
        for r in Rs:
            lengthsS = []
            for ix in idxs :
                KA, KB, KAB, XA, XB, _, SB, _, V_SB = data[ix]
                KA = KA.squeeze()
                KB = KB.squeeze()
                KAB = KAB.squeeze()
                KA_combined = combine_kernels(KA, torch.DoubleTensor(alpha), gamma1)  
                KB_combined = combine_kernels(KB, torch.DoubleTensor(alpha), gamma1)  
                KAB_combined = combine_kernels(KAB, torch.DoubleTensor(alpha), gamma1)
                group = data.groups[ix]
                subset = data.get_subset_df(group, 'B')
                mmd = MMD_comp( KB_combined, KA_combined, KAB_combined, 1./torch.ones(KB_combined.shape[0]), 1./torch.ones(KA_combined.shape[0]), lambdaa)
                           
                lengths = subset["num_words"].values
                S, cost = greedy_maximize(mmd, budget = BUDGET, 
                            costs = copy(lengths), r = r, verbose = False )
                lengthsS.append(len(S))
                write_result(group, 'B', subset, S, alpha_seq, r )
        logger.info("model:{}, |S|:{:.2g}, #topics:{}".format((gamma1, lambdaa, alpha_seq, r), np.mean(lengthsS), len(idxs) ))

def infer_all():
    pool = mp.Pool(6)
    jobs = []
    
    for lambdaa, gamma1 in product(Lambdas, Gammas):
        for MODE in ["test", "val"]:
            jobs.append( pool.apply_async (infer, (gamma1, lambdaa, MODE ) ))

    for job in jobs:    
        job.get()
    pool.close()

if __name__ == "__main__":
    infer_all()