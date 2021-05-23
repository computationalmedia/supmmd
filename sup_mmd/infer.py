import torch
from sup_mmd.data import MMD_Dataset
import sys, json, re, os, glob
from commons.utils import get_logger
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sup_mmd.model import LinearModel1, mmd_loss_pq, LinearModelComp1, mmd_loss_pq_comp
from sup_mmd.model import MMD, MMD_comp
from submodular.maximize import greedy as greedy_maximize
import pandas as pd
from copy import copy
from sup_mmd.functions import softmax, nz_median_dist, combine_kernels
import multiprocessing as mp

logger = get_logger("Infer")
Rs = [ 0.0, 0.001, 0.003, 0.01]
GPU_MODE = False

pattern_generic = re.compile(r'(mmdpq?)(_)(.?)(duc03|duc04|tac08|tac09)-([AB])_([xc])\.(\d+|x)_g(\d\.?\d*)_b(\d\.?\d*)_a(\d\.?\d*)(_)SF(b|x)(k|x)(c|x)') 
pattern_update = re.compile(r'(mmdpq?)-comp([01])\.(lin1|lin2)_(tac08|tac09)-([AB])_([xc])\.(\d+|x)_g(\d\.?\d*)_b(\d\.?\d*)_a(\d+)_l(\d\.?\d*)_SF(b|x)(k|x)(c|x)')   

TARGET_NAME = "y_hm_0.4"
# TARGET_NAME = "y_R2_0.0"
BUDGET = 125
CACHE_ROOT = "./data/"

def infer(model_root, model_path = None, MODE = None):
    model_path = model_path or sys.argv[1]
    MODE = MODE or sys.argv[2]
    logger.info("processing model:{}".format(model_path))
    assert MODE in {"train", "val", "test"}

    model_file = model_path.split("/")[-1]
    s = pattern_generic.search(model_file)
    generic = True ## MMD() or MMD() - lambda*MMD()
    if not s:
        s = pattern_update.search(model_file)
        if not s:
            logger.error("pattern not matched with both generic/update regexes, quitting " + model_file)
            return
        generic = False

    name = s.group(0)
    loss_name = s.group(1)
    assert loss_name in {"mmdpq"}
    train_dataset = s.group(4).lower()
    assert train_dataset in {"duc03", "duc04", "tac08", "tac09"}
    set_ = s.group(5)
    assert set_ in {"A", "B"}
    if not generic or set_ == "B":
        assert train_dataset in {"tac08", "tac09"}
    compress = s.group(6) == "c"
    split_seq = s.group(7)
    
    if MODE in {"train", "test"}:
        if split_seq != "x":
            return
    if MODE == "val":    
        try:
            split_seq = int(split_seq)
        except:
            logger.warning("val mode not applicable for model w/o val")
            return

    gamma1 = float(s.group(8))
    beta = float(s.group(9))
    alpha_seq = s.group(10)
    lambdaa = 0.0

    lambdaa, diff, model_name = s.group(11), s.group(2), s.group(3)
    if not generic:
        assert set_ == "B"
        lambdaa = float(s.group(11))
        diff = int(s.group(2))
        model_name = s.group(3)
    
    BOOST_FIRST = s.group(12) == "b"
    KEYWORDS = s.group(13) == "k"
    comp_feats = s.group(14) =="c"
    
    logger.debug((name, list(s.groups()), (loss_name, diff, model_name, train_dataset, set_, compress, split_seq, gamma1, beta, alpha_seq, lambdaa, BOOST_FIRST, KEYWORDS, comp_feats) ) )
    if MODE in { "train", "val" }:
        dataset = train_dataset
    else:
        dataset = {
            "duc03": "duc04", 
            "duc04": "duc03",
            "tac08": "tac09",
            "tac09": "tac08"
        }[train_dataset]

    dataset_name = "{}_{}".format(dataset, TARGET_NAME)

    logger.debug("loading data from " + dataset_name)
    data = MMD_Dataset.load(dataset_name, CACHE_ROOT, compress = compress)
    SURF_IDXS = data.surf_idxs(keywords = KEYWORDS, boost_first = BOOST_FIRST, comp = ( comp_feats and set_ == "B" ) )
    logger.info("surf feats: {}".format(
        ",".join( np.array(data.surf_names)[SURF_IDXS] )
    ))

    logger.debug("loading model from " + model_path)
    try:
        model, alpha, train_idxs, val_idxs, epochs = LinearModel1.load(len(SURF_IDXS), model_path)
    except:
        model, alpha, train_idxs, val_idxs, epochs = LinearModelComp1.load(len(SURF_IDXS), len(SURF_IDXS), model_path)

    if MODE == "train":
        idxs = train_idxs
    elif MODE == "val":
        idxs = val_idxs
    elif MODE == "test":
        idxs = np.arange( len(data) ).tolist()

    root = "{}/rouge_{}".format(model_root, MODE)
    if not os.path.exists(root):
        try:
            os.makedirs(root)
        except:
            pass
    logger.debug("Dataset and model loaded, begin inference with #topics={}, generic?={}".format( len(idxs), generic ))

    def write_result(group, set_, subset, S, r, **kwargs):
        set_ = ("-%s"%set_) if len(data.sets) > 1 else ""
        fname = "%s%s.M.100.X.%s_r%.2g_ep%d"%( group.upper(), set_, name, r, epochs)
        sents = subset["sent"].iloc[S].values
        path_write = "{}".format(root)
        # path_write = "{}/gamma_{}_beta_{}".format(root, gamma1, beta)
        if not os.path.exists(path_write):
            try:
                os.makedirs(path_write)
            except:
                pass
        with open(path_write + "/" + fname, "w") as fp:
            fp.write("\n".join(sents))
        return

    for r in Rs:
        if GPU_MODE:
            model.cuda()
        lengthsS = []
        for ix in idxs :
            if generic:
                if train_dataset in ["duc03", "duc04"]:
                    K, X, _, _ = data[ix]
                elif train_dataset in ["tac08", "tac09"]:
                    if set_ == "A":
                        # logger.info("A")
                        K, _, _, X, _, _, _, _, _ = data[ix]
                    else:
                        # logger.info("B")
                        _, K, _, _, X, _, _, _, _ = data[ix]
                
                K, X = K.squeeze(), X.squeeze()[:, SURF_IDXS]
                fg = model.forward( X )[0]
                K_combined = combine_kernels(K, alpha, gamma1) 
                mmd = MMD(K_combined, fg)
            else:
                KA, KB, KAB, XA, XB, _, _, _, _ = data[ix]
                KA, XA = KA.squeeze(), XA.squeeze()[:, SURF_IDXS]
                KB, XB = KB.squeeze(), XB.squeeze()[:, SURF_IDXS]
                KAB = KAB.squeeze()
                
                fA, fB = model.forward( XA, XB )
                KA_combined = combine_kernels(KA, alpha, gamma1)  
                KB_combined = combine_kernels(KB, alpha, gamma1)  
                KAB_combined = combine_kernels(KAB, alpha, gamma1)
                mmd = MMD_comp( KB_combined, KA_combined, KAB_combined, fB, fA, lambdaa = lambdaa, diff = diff)

            group = data.groups[ix]            
            subset = data.get_subset_df(group, set_ )

            lengths = subset["num_words"].values
            keys = None
            if compress:
                keys = [int(sid.split("-")[0]) for sid in subset["sent_id"]]
            S, cost = greedy_maximize(mmd, budget = BUDGET, 
                        costs = copy(lengths), r = r, verbose = False, keys = keys)
            lengthsS.append(len(S))
            write_result(group, set_, subset, S, r )
    logger.info("{}-{}[{}]:{}, |S|:{:.2g}, #topics:{}".format(
        dataset_name, set_, MODE, 
        name, np.mean(lengthsS), len(idxs) 
    ))

def infer_all():
    pool = mp.Pool(32)
    jobs = []
    model_root = sys.argv[1]
    for model_path in glob.glob('%s/states/*.net'%model_root, recursive=True):
        for MODE in ["train", "test", "val"]:
            jobs.append( pool.apply_async (infer, (model_root, model_path, MODE ) ))

    for job in jobs:    
        job.get()
    pool.close()

if __name__ == "__main__":
    infer_all()