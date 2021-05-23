import torch
from sup_mmd.data import MMD_Dataset
import sys, json, os, shutil, traceback, operator
from commons.utils import get_logger, apply_file_handler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sup_mmd.model import LinearModel1, mmd_loss_pq
import multiprocessing as mp
from copy import copy
from itertools import product, chain
from sup_mmd.functions import nz_median_dist, combine_kernels
from pyhocon import ConfigFactory
from sup_mmd.model import EarlyStopping
from numpy.random import RandomState
from math import ceil
import torch_optimizer as optim

np.random.seed(19)

GPU_MODE = False
conf = ConfigFactory.parse_file( sys.argv[1] )
TRAIN_DATASET = conf.get( "app.train_dataset" ).lower()
assert TRAIN_DATASET.lower() in ["duc03", "duc04", "tac08", "tac09"]

TEST_DATASET = {
    "duc03": "duc04", 
    "duc04": "duc03",
    "tac08": "tac09",
    "tac09": "tac08"
}[TRAIN_DATASET.lower()]

ROOT = conf.get("app.ROOT")
RUN_ID = conf.get("app.runID")
N_JOBS = int(conf.get("app.N_JOBS"))
COMPRESS = conf.get("app.compress", False)
STORE_PATH = "%s/%s/"%( ROOT, RUN_ID ) 
CV_repeat = conf.get("app.CV", 0)
early_stop_reduce = conf.get("param.early_stopping.reduce", "max")
assert early_stop_reduce in {"max", "avg"}

EARLY_DELTA = conf.get("param.early_stopping.delta", 1e-5 )
EARLY_PATIENCE = conf.get("param.early_stopping.patience", 4 )
optimizer_name = conf.get("param.optimizer.name")
optim_args = json.loads(conf.get("param.optimizer.args", "{}"))
assert optimizer_name.lower() in {"adam", "adamw", "sgd", "lbfgs", "adagrad", "rmsprop", "yogi"}
set_ = conf.get("app.set", 'A')
comp_feats = conf.get("app.comp_feats", False) ## might be useful for set B (novelity fetures)

if TRAIN_DATASET in {"duc03", "duc04"}:
    assert set_ == 'A'
elif TRAIN_DATASET in {"tac08", "tac09"}:
    assert set_ in { 'B', 'A' }

OPTIM = {
    "lbfgs": torch.optim.LBFGS,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "adagrad": torch.optim.Adagrad,
    "rmsprop": torch.optim.RMSprop,
    "adamw": torch.optim.AdamW,
    "yogi": optim.Yogi
}[optimizer_name.lower()]

batch_size = conf.get("param.batch_size", None)
assert batch_size in {1, 2, 4, 8, None}, "batch_size in {1, 2, 4, 8, None} only allowed"

logger = get_logger("%s"%RUN_ID)
logger.info("#procs:%d, optimizer:%s"%( N_JOBS, optimizer_name ))
logger.info(conf)

EPOCHS = conf.get("param.EPOCHS")
assert N_JOBS > 0 and EPOCHS >= 5

try:
    shutil.rmtree(STORE_PATH)
except:
    pass
if not os.path.exists(STORE_PATH + "/states/"):
    os.makedirs(STORE_PATH + "/states/", exist_ok = True)
    os.makedirs(STORE_PATH + "/logs/", exist_ok = True)
    shutil.copy2( sys.argv[1], STORE_PATH )

GAMMAs1 = conf.get("param.GAMMAs")
BETAs = conf.get("param.BETAs")
Alphas = conf.get("param.ALPHAs")

TARGET_NAME = conf.get("app.target_name", "y_hm_0.4")
assert TARGET_NAME in {"y_hm_0.4", "y_R2_0.0"}
train_name = "{}_{}".format(TRAIN_DATASET, TARGET_NAME)
test_name = "{}_{}".format(TEST_DATASET, TARGET_NAME)

dataset_train = MMD_Dataset.load(train_name, "%s/data/"%( ROOT ), compress = COMPRESS  )
dataset_test = MMD_Dataset.load(test_name, "%s/data/"%( ROOT ), compress = COMPRESS  )
logger.info("TRAIN_DATASET:{}".format(str(dataset_train)))
logger.info("TEST_DATASET:{}".format(str(dataset_test)))

## checking for consistency between datasets
assert dataset_train.surf_names == dataset_test.surf_names
assert dataset_train.kernel_names == dataset_test.kernel_names
assert dataset_train.target_name == dataset_test.target_name

KEYWORDS = conf.get("app.keywords", False)
BOOST_FIRST = conf.get("app.boost_first", True)

SURF_IDXS = dataset_train.surf_idxs(
    keywords = KEYWORDS, boost_first = BOOST_FIRST, 
    comp = ( set_ == "B" and comp_feats ) 
)

if GPU_MODE:
    dataset_train.cuda()
    dataset_test.cuda()
else:
    torch.set_num_threads(1)

logger.info("surf feats: {}".format(
    ",".join( np.array(dataset_train.surf_names)[SURF_IDXS] )
))

## train one model for given hyperparams
def run_train_split(name, train_idxs, val_idxs, gamma1, beta, alpha, epochs = EPOCHS ):
    ## logging
    _logger = get_logger("TRAIN")
    apply_file_handler(_logger, STORE_PATH + "/logs/" + name + ".log")
    logger.info("begin training:{} #rows: train=>{}, #train:{}, #val:{}".format(
        name, len(dataset_train), len(train_idxs), len(val_idxs)
    ))
    _logger.info("begin training:{} #rows: train=>{}, #train:{}, #val:{}, optimizer:{}".format(
        name, len(dataset_train), len(train_idxs), len(val_idxs), optimizer_name
    ))
    loss = mmd_loss_pq
    writer = SummaryWriter('{}/runs/{}'.format(STORE_PATH, name))
    
    ## initialize model
    MODEL_PATH = '{}/states/{}.net'.format(STORE_PATH, name  )
    model1 = LinearModel1( d_sf = len(SURF_IDXS) ).double()

    ## initialize optimizer
    optimizer = OPTIM( model1.parameters(), **optim_args )

    _logger.info("W0: wt={}, b={:.4g}, a={}".format(
        model1.lin_sf.weight.data.numpy(), model1.lin_sf.bias.data.numpy()[0],
        alpha.data.numpy()
    ))
    _logger.info((train_idxs, val_idxs))
    _logger.info("train groups:{}".format(",".join(np.array(dataset_train.groups)[train_idxs])))
    if len(val_idxs) > 0:
        _logger.info("val groups:{}".format(",".join(np.array(dataset_train.groups)[val_idxs])))
        ## early stopping
        es = EarlyStopping( min_delta = EARLY_DELTA, patience = EARLY_PATIENCE )
    else:
        _logger.info("RETRAINING MODE with epochs:{}".format(epochs))
    
    epoch = 0

    def step(data, idxs, corr = False):
        loss_accum = torch.DoubleTensor([0.0])
        if GPU_MODE:
            loss_accum = loss_accum.cuda() 
        corr_accum = []
        for idx in idxs :
            if TRAIN_DATASET in ["duc03", "duc04"]:
                K, X, S, V_S = data[idx]
            elif TRAIN_DATASET in ["tac08", "tac09"]:
                if set_ == 'A':
                    K, _, _, X, _, S, _, V_S, _ = data[idx]
                else:
                    _, K, _, _, X, _, S, _, V_S = data[idx]
           
            K, X = K.squeeze(), X.squeeze()[:, SURF_IDXS]
            f = model1.forward( X )[0]
            # print(fA)
            K_combined = combine_kernels(K, alpha, gamma1)           
            loss_val = loss(f, K_combined, S, V_S)
            loss_accum += loss_val
            if corr:
                subset = data.get_subset_df(data.groups[idx], set_)
                rouge2 = subset['R2.R'].values / subset['num_words'].values
                corr_accum.append( [loss_val.item(), np.corrcoef(f.detach().numpy(),rouge2)[0, 1] ] )
        loss_accum = loss_accum / len(idxs)
        return loss_accum, np.array(corr_accum)

    while epoch < epochs:
        _logger.debug("Epoch %d"%( epoch+1))
        ### training part in mini batches
        # np.random.shuffle(train_idxs)
        num_batches = (len(train_idxs) // batch_size) if batch_size is not None else 1
        batches = np.array_split(np.random.permutation(train_idxs), num_batches)
        
        for batch_idxs in batches:
            def closure():
                optimizer.zero_grad()
                train_loss, _ = step(dataset_train, batch_idxs)
                train_loss += beta * model1.reg()
                train_loss.backward()
                return train_loss
            optimizer.step(closure)
        
        ## for logging
        train_loss, train_corr = step(dataset_train, train_idxs, True)
        _logger.info("Train [{:02d}], loss: {:.6g}, corr: {:.3g}".format(
            epoch+1, train_loss.item(), np.mean(train_corr[:, 1]) ) )
        writer.add_scalar('avg_loss/train', train_loss.item(), epoch + 1)
        writer.add_scalar('corr/train', np.mean(train_corr[:, 1]), epoch +1)
        # writer.add_histogram('corr.hist/train', train_corr[:, 1], epoch +1)
        # writer.add_histogram('mmd.hist/train', train_corr[:, 0], epoch +1)
        writer.add_scalar('reg', np.log(model1.reg().item()), epoch + 1)

        ### Validation part
        if len(val_idxs) > 0:
            val_loss, val_corr = step(dataset_train, val_idxs, True)
            _logger.info("Val__ [{:02d}], loss: {:.6g}, corr: {:.3g}".format(
            epoch+1, val_loss.item(), np.mean(val_corr[:, 1]) ) )
            writer.add_scalar('avg_loss/val', val_loss.item(), epoch + 1)
            writer.add_scalar('corr/val', np.mean(val_corr[:, 1]), epoch +1)
            # writer.add_histogram('corr.hist/val', val_corr[:, 1], epoch +1)
            # writer.add_histogram('mmd.hist/val', val_corr[:, 0], epoch +1)
    
        ### test part, just for logging, not used in any decision making
        test_loss, test_corr = step(dataset_test, np.arange(len(dataset_test)), True)
        _logger.info("Test_ [{:02d}], loss: {:.6g}, corr: {:.3g}".format(
            epoch+1, test_loss.item(), np.mean(test_corr[:, 1]) ) )
        
        writer.add_scalar('avg_loss/test', test_loss.item(), epoch + 1)
        writer.add_scalar('corr/test', np.mean(test_corr[:, 1]), epoch +1)
        # writer.add_histogram('corr.hist/test', test_corr[:, 1], epoch +1)
        # writer.add_histogram('mmd.hist/test', test_corr[:, 0], epoch +1)
        writer.flush()

        ## early stopping if val set is available
        epoch += 1
        if len(val_idxs) > 0:
            if es.step( torch.DoubleTensor([val_loss]) ):
                _logger.info("early stopping at {} epoch".format(epoch))
                break
            _logger.info( "EARLY_STOPPING epoch:{}, {}".format( epoch, str(es)) )
    ### save model
    _logger.info("Saving model {}".format( name ))
    model1.save(alpha, train_idxs, val_idxs, epoch, MODEL_PATH)

    wts = ", ".join( ["{}:{:.4g}".format(n, v) for n, v in zip(np.array(dataset_train.surf_names)[SURF_IDXS], 
            model1.lin_sf.weight.data.numpy().squeeze())
    ] )
    _logger.info("Wn: wt={}, b={:.4g}, a={}".format(
        wts, 
        model1.lin_sf.bias.data.numpy()[0], 
        alpha.data.numpy()
    ))
    logger.info("Wn[{}]: wt={}, b={:.4g}, a={}, epochs:{}".format(
        name, 
        wts, 
        model1.lin_sf.bias.data.numpy()[0], 
        alpha.data.numpy(),
        epoch
    ))
    writer.close()
    return name, epoch

def run_train(name, gamma1, beta, alpha):
    # val_size = ceil ( len(dataset_train) * VAL_SPLIT )
    if CV_repeat >= 3:
        collected_epochs = []
        idxs = np.arange(len(dataset_train))
        splits = np.array_split(idxs, CV_repeat)
        for i in range(len(splits)):
            _name = name%i
            val_idxs = splits[i].squeeze()
            mask = np.ones(CV_repeat, dtype = bool)
            mask[i] = False
            train_idxs = list(chain(*np.array(splits)[mask]))
            
            # print(train_idxs, val_idxs)
            # r = np.random.RandomState(split)
            # r.shuffle(idxs)
            # train_idxs, val_idxs = idxs[ :val_size ], idxs[ val_size: ]
            _name2, epoch = run_train_split(_name, train_idxs, val_idxs, gamma1, beta, alpha, EPOCHS )
            collected_epochs.append( epoch )
        logger.info("{}, #epochs:{}".format( name, collected_epochs ))
        if early_stop_reduce == "avg":
            retrain_epochs = ceil(sum(collected_epochs) / CV_repeat)
        elif early_stop_reduce == "max": 
            retrain_epochs = max(collected_epochs)
    else:
        retrain_epochs = EPOCHS
    logger.info("RETRAINING {} with {} epochs".format(name%"x", retrain_epochs))
    run_train_split(name%"x", np.arange(len(dataset_train)), [], gamma1, beta, alpha, retrain_epochs )
    logger.info("Completed {}".format(name%"x"))

def main():
    # manager = mp.Manager()
    pool = mp.Pool(N_JOBS)
    jobs = []
    
    ## run on grid
    try:
        for gamma1, beta in product(GAMMAs1, BETAs):
            for alpha_seq, alpha in enumerate(Alphas):
                assert len(dataset_train.kernel_names) == len(alpha)
                name = "mmdpq_{}-{}_{}.%s_g{}_b{}_a{}_SF{}{}{}".format(
                    TRAIN_DATASET, 
                    set_,
                    "c" if COMPRESS else "x",
                    gamma1, beta, alpha_seq,
                    "b" if BOOST_FIRST else "x",
                    "k" if KEYWORDS else "x",
                    "c" if comp_feats else "x",
                )
                
                if GPU_MODE:
                    model1.cuda()
                logger.info(name)
                job = pool.apply_async( run_train, (name, gamma1, beta, torch.DoubleTensor(alpha)))
                jobs.append( job )

        logger.info("running {} jobs".format(len(jobs)))
        # jobs.sort(key=operator.itemgetter(0))
        for job in jobs:    
            job.get()
        pool.close()

    except Exception as ex:
        logger.error(ex)
        traceback.print_exc()
        raise ex
    logger.info("DONE")

if __name__ == "__main__":
    main()