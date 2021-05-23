import torch
from sup_mmd.data import MMD_Dataset
import sys, json, re, os, glob, shutil
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
import pandas as pd
from sup_mmd.compress import SentenceCompressor

logger = get_logger("Infer")
GPU_MODE = False

pattern_generic = re.compile(r'(mmdpq?)(_)(.?)(duc03|duc04|tac08|tac09)-([AB])_([xc])\.(\d+|x)_g(\d\.?\d*)_b(\d\.?\d*)_a(\d\.?\d*)(_)SF(b|x)(k|x)(c|x)') 
pattern_update = re.compile(r'(mmdpq?)-comp([01])\.(lin1|lin2)_(tac08|tac09)-([AB])_([xc])\.(\d+|x)_g(\d\.?\d*)_b(\d\.?\d*)_a(\d+)_l(\d\.?\d*)_SF(b|x)(k|x)(c|x)')   

TARGET_NAME = "y_hm_0.4"
BUDGET = 150 ##aftre compression, it will be less
#### ROUGE eval truncates, so > 100 words will be truncated

CACHE_ROOT = "./data/"
# NOTE: SRL model available at: https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz
SRL_PATH = "models/srl-model-2018.05.25.tar.gz"

model_path = sys.argv[1]
logger.info("processing model:{}".format(model_path))
r = float(sys.argv[2])
COMPRESSED = sys.argv[3] == "1"
if COMPRESSED:
    logger.warning("compressed mode on, this will be slow")
    sc = SentenceCompressor(SRL_PATH)
   
def write_result(group, set_, subset, S, name, r, epochs, root, **kwargs):
    sents = subset["sent"].iloc[S].values
    fname = "%s%s.M.100.X.%s_r%.2g_ep%d_x"%( group.upper(), set_, name, r, epochs)
    with open(root + "/summaries/" + fname, "w") as fp:
        fp.write("\n".join(sents))
    
    if COMPRESSED:
        fname = "%s%s.M.100.X.%s_r%.2g_ep%d_c"%( group.upper(), set_, name, r, epochs)
        sents = [ min(sc.compress(s), key = len) for s in sents ]
        with open(root + "/summaries/" + fname, "w") as fp:
            fp.write("\n".join(sents))
    return

def infer():
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
    
    if split_seq != "x":
        logger.warning("please supply retrained model")

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

    idxs = np.arange( len(data) ).tolist()

    root = "./{}_{}/".format( dataset, set_ )

    if not os.path.exists(root + "summaries"):
        try:
            os.makedirs(root + "summaries")
        except:
            pass
    shutil.copy2( model_path, root )

    logger.debug("Dataset and model loaded, begin inference with #topics={}, generic?={}".format( len(idxs), generic ))

    if GPU_MODE:
        model.cuda()
    
    lengthsS = []
    jobs = []
    pool = mp.Pool(24)
    dfs = []

    for ix in idxs :
        group = data.groups[ix]            
        subset = data.get_subset_df(group, set_ )
        write_df = subset[["position", "doc_sents", "sent_id", "group", "set", "doc_id", "num_words", 
                    "R1.R", "R1.P", "R2.R", "R2.P", "nouns", "prpns", "target"]]
        surf_names = np.array([s.strip() for s in data.surf_names])[SURF_IDXS]

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
            KA, KB, KAB, XA, X, _, _, _, _ = data[ix]
            KA, XA = KA.squeeze(), XA.squeeze()[:, SURF_IDXS]
            KB, X = KB.squeeze(), X.squeeze()[:, SURF_IDXS]
            KAB = KAB.squeeze()
            
            fA, fg = model.forward( XA, X )
            KA_combined = combine_kernels(KA, alpha, gamma1)  
            KB_combined = combine_kernels(KB, alpha, gamma1)  
            KAB_combined = combine_kernels(KAB, alpha, gamma1)
            mmd = MMD_comp( KB_combined, KA_combined, KAB_combined, fg, fA, lambdaa = lambdaa, diff = diff)
            write_df["nf"] = X[:, np.where(surf_names=="nf")[0]] #normalised

        write_df["lexrank"] = X[:, np.where(surf_names=="lexrank")[0]].numpy() + 1.0
        write_df["tfisf"] = X[:, np.where(surf_names=="tfisf")[0]].numpy() ## normalised
        write_df["btfisf"] = X[:, np.where(surf_names=="btfisf")[0]].numpy() #normalised
        write_df["scores"] = softmax(fg.detach().numpy()) * len(subset)

        dfs.append(write_df)
        lengths = subset["num_words"].values
        keys = None
        if compress:
            keys = [int(sid.split("-")[0]) for sid in subset["sent_id"]]
        S, cost = greedy_maximize(mmd, budget = BUDGET, 
                    costs = copy(lengths), r = r, verbose = False, keys = keys)
        lengthsS.append(len(S))

        jobs.append( pool.apply_async(write_result, (group, ("-%s"%set_) if len(data.sets) > 1 else "", subset, S, name, r, epochs, root )))

    logger.info("{}-{}:{}, |S|:{:.2g}, #topics:{}".format(
        dataset_name, set_, 
        name, np.mean(lengthsS), len(idxs) 
    ))
    pd.concat(dfs).to_csv("{}/sents.csv".format(root), index = False, float_format='%.6g')
    for job in jobs:    
        job.get()
    pool.close()
    logger.info("Done")

# def infer_all():
    
if __name__ == "__main__":
    infer()
