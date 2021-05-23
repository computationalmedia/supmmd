import torch
from submodular.base import Function
import torch.nn.functional as F
import numpy as np
from copy import copy
from scipy.stats import describe
from commons.utils import get_logger
logger = get_logger("LOSS")

def weights_init(m):
    torch.manual_seed(19)
    classname = m.__class__.__name__
    # m.weight.data.fill_(0.05)
    torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.05)
    # m.weight.data.uniform_(-0.1, 0.1)
    m.bias.data.fill_(0.05)

class LinearModel1(torch.nn.Module):
    """ 
        linear model of importance
    """
    def __init__(self, d_sf, **kwargs):
        super(LinearModel1, self).__init__()
        self.lin_sf = torch.nn.Linear(d_sf, 1, bias = True)
        self.lin_sf.apply(weights_init)
        self.name = "lin1"

    def forward(self, *X):
        return tuple( self.lin_sf(_X).squeeze() for _X in X )

    def reg(self):
        _reg = 0
        for param in self.lin_sf.parameters():
            _reg += (param ** 2).sum()
        return _reg
    
    # def regL(self, f, K):
    #     p = F.softmax(f, dim = 0)

    
    ### while saving also save alpha: kenel weights
    ### and train/val idxs from frain datasets
    ### idxs corresponds to groups in datasets
    ### and number of epochs the model is trained
    def save(self, alpha, train_idxs, val_idxs, epochs, MODEL_PATH):
        data = {
            "model_dict": self.state_dict(),
            "alpha": alpha.data.numpy(),
            "train_idxs": train_idxs,
            "val_idxs": val_idxs,
            "epochs": epochs
        }
        torch.save( data, MODEL_PATH  )
    
    @staticmethod
    def load(d_sf, model_path):
        model = LinearModel1( d_sf = d_sf )
        state = torch.load(model_path)
        # print(state)
        model.load_state_dict(state["model_dict"])
        model = model.double()
        alpha = torch.DoubleTensor( state["alpha"] )
        train_idxs = state["train_idxs"]
        val_idxs = state["val_idxs"]
        epochs = state["epochs"]
        return model, alpha, train_idxs, val_idxs, epochs

class LinearModelComp1(torch.nn.Module):
    """
        Linear model of importance for comparative summarisation
        two linear models for summarising group and comparing group
    """
    def __init__(self, d_sfA, d_sfB, **kwargs):
        super(LinearModelComp1, self).__init__()
        self.lin_sfA = torch.nn.Linear(d_sfA, 1, bias = True)
        self.lin_sfB = torch.nn.Linear(d_sfB, 1, bias = True)
        self.lin_sfA.apply(weights_init)
        self.lin_sfB.apply(weights_init)
        self.name = "lin2"

    def forward(self, XA, XB ):
        return self.lin_sfA(XA).squeeze(), self.lin_sfB(XB).squeeze()

    def reg(self):
        _reg = 0
        for param in self.parameters():
            _reg += (param ** 2).sum()
        return _reg
    
    def regA(self):
        _reg = 0
        for param in self.lin_sfA.parameters():
            _reg += (param ** 2).sum()
        return _reg
    
    def regB(self):
        _reg = 0
        for param in self.lin_sfB.parameters():
            _reg += (param ** 2).sum()
        return _reg

    ###
    def save(self, alpha, train_idxs, val_idxs, epochs, MODEL_PATH):
        data = {
            "model_dict": self.state_dict(),
            "alpha": alpha.data.numpy(),
            "train_idxs": train_idxs,
            "val_idxs": val_idxs,
            "epochs": epochs
        }
        torch.save( data, MODEL_PATH  )
    
    @staticmethod
    def load(d_sfA, d_sfB, model_path):
        model = LinearModelComp1( d_sfA = d_sfA, d_sfB = d_sfB )
        state = torch.load(model_path)
        # print(state)
        model.load_state_dict(state["model_dict"])
        model = model.double()
        alpha = torch.DoubleTensor( state["alpha"] )
        train_idxs = state["train_idxs"]
        val_idxs = state["val_idxs"]
        epochs = state["epochs"]
        return model, alpha, train_idxs, val_idxs, epochs

def mmd_loss_pq_unbiased(f, K, S, V_S, **kwargs):
    fV = torch.index_select(f, 0, V_S)
    fS = torch.index_select(f, 0, S)
    pV = F.softmax(fV, dim = 0)
    qS = F.softmax(fS, dim = 0)
    K_VV = torch.index_select( torch.index_select(K, 0, V_S), 1, V_S)
    K_VS = torch.index_select( torch.index_select(K, 0, V_S), 1, S)
    K_SS = torch.index_select( torch.index_select(K, 0, S), 1, S)
    ## adjusting for unbiased estimator
    o1 = torch.dot(pV, torch.mv( K_VV, pV )) 
    o2 = torch.dot(pV, torch.mv( K_VS, qS ))
    o3 = torch.dot(qS, torch.mv( K_SS , qS )) 
    return torch.sqrt(o1 - 2.0 * o2 + o3) 

def mmd_loss_pq(f, K, S, V_S, **kwargs):
    # fV = torch.index_select(f, 0, V_S)
    fS = torch.index_select(f, 0, S)
    pV = F.softmax(f, dim = 0)
    qS = F.softmax(fS, dim = 0)
    K_VS = torch.index_select( K, 1, S)
    K_SS = torch.index_select( torch.index_select(K, 0, S), 1, S)
    ## adjusting for unbiased estimator
    o1 = torch.dot(pV, torch.mv( K, pV )) 
    o2 = torch.dot(pV, torch.mv( K_VS, qS ))
    o3 = torch.dot(qS, torch.mv( K_SS , qS )) 
    return torch.sqrt(o1 - 2.0 * o2 + o3)
    # return o1 - 2.0 * o2 + o3

def mmd_loss_pq_comp(fB, fA, KB, KA, KAB, SB, V_SB, lambdaa = 0.1, diff = 1, **kwargs):
    assert diff in {0, 1}
    fSB = torch.index_select(fB, 0, SB)
    pVB = F.softmax(fB, dim = 0)
    pVA = F.softmax(fA, dim = 0)
    qS = F.softmax(fSB, dim = 0)
    
    K_VSB = torch.index_select( KB, 1, SB)
    KA_SB = torch.index_select( KAB, 1, SB)
    K_SS = torch.index_select( torch.index_select(KB, 0, SB), 1, SB)

    o1 = torch.dot(pVB, torch.mv( KB, pVB ))
    o2 = torch.dot(pVB, torch.mv( K_VSB, qS ))
    o3 = torch.dot(qS, torch.mv( K_SS , qS ))

    c1 = torch.dot(pVA, torch.mv( KA, pVA )) 
    c2 = torch.dot(pVA, torch.mv( KA_SB, qS ))

    mmd2_pos = o1 - 2.0 * o2 + o3
    mmd2_neg = c1 - 2.0 * c2 + diff * o3
    return ( torch.sqrt(mmd2_pos) - lambdaa * torch.sqrt(mmd2_neg) )
    # return mmd2_pos - lambdaa * mmd2_neg

def mmd_loss_pq_compA(fB, fA, KB, KA, KAB, SB, V_SB, lambdaa = 0.1, diff = 1, **kwargs):
    assert diff in {0, 1}
    fSB = torch.index_select(fB, 0, SB)
    # pVB = F.softmax(fB, dim = 0)
    pVA = F.softmax(fA, dim = 0)
    qS = F.softmax(fSB, dim = 0)
    
    K_VSB = torch.index_select( KB, 1, SB)
    KA_SB = torch.index_select( KAB, 1, SB)
    K_SS = torch.index_select( torch.index_select(KB, 0, SB), 1, SB)

    o3 = torch.dot(qS, torch.mv( K_SS , qS ))
    c1 = torch.dot(pVA, torch.mv( KA, pVA )) 
    c2 = torch.dot(pVA, torch.mv( KA_SB, qS ))

    mmd2_neg = c1 - 2 * c2 + diff * o3
    # return ( - lambdaa * mmd2_neg )
    return ( - lambdaa * torch.sqrt( mmd_neg ) )

class MMD(Function):
    '''
        F(S+s) - F(S) for MMD (Kim et al NIPS 2016)
        = AvgDiv + 2 * AvgCover

    '''
    def __init__(self, K, f, **kwargs):
        super().__init__(len(K))
        self._f = f.detach().squeeze()
        # print(describe(self._f.numpy()))
        self.pV = F.softmax( self._f, dim = 0 ).numpy()
        # print(describe(self.pV))
        # print(self.pV)
        self._K = K.numpy()
        self._r = np.dot(self._K, self.pV)
        self._o1 = np.dot(self.pV, self._r)
        self._C = 0.0 #np.max(self._f.numpy())
        assert self._K.shape == ( len(self._r), len(self._r) )

    def add2S(self, s):
        return super().add2S(s)

    def F(self, S = None, **kwargs):
        S_ = self._S if S is None else copy(S)
        lenS = (self._nS if S is None else np.sum(S))
        S_ = np.where(S_ == 1)[0]
        mmd = self._o1
        if lenS > 0:
            # print(S, self._K.shape )
            qS = F.softmax(self._f[S_].squeeze(), dim = 0).numpy()
            K_SS = self._K[S_, :][:, S_].squeeze()
            mmd -= 2.0 * np.dot(self._r[S_], qS)
            mmd += np.dot(qS, np.dot( K_SS, qS ))
        return -mmd
    
    def marginal_gain(self, S = None, **kwargs):
        S_ = self._S if S is None else copy(S)
        lenS = (self._nS if S is None else np.sum(S))
        S_ = np.where(S_ == 1)[0]
        e_fS = 0.0
        term1, term3, term4 = 0.0, 0.0, 0.0
        e_fsk = np.exp(self._f.numpy() - self._C)
        if lenS > 0:
            qS = F.softmax(self._f[S_].squeeze(), dim = 0).numpy()
            e_fS = np.sum( np.exp(self._f[S].numpy() - self._C) )
            term1 = np.dot(self._r[S_], qS) * (- e_fsk / (e_fsk + e_fS) )
            
            term3 = np.dot(qS, np.dot(self._K[S_, :][:, S_].squeeze(), qS)) * \
                            (e_fS ** 2 - (e_fS + e_fsk) ** 2) / (e_fS + e_fsk ) ** 2
            tmp = (e_fsk / (e_fS + e_fsk)) * 1./ (1. + e_fsk / e_fS).squeeze()
            term4 = np.dot(self._K[:, S_], qS).squeeze() * tmp 
            # print(term1.shape, term3.shape, term4.shape)
        # print(type(self._r), type(e_fsk))
        term2 = self._r * e_fsk / ( e_fsk + e_fS )
        term5 = ( e_fsk / (e_fS + e_fsk) ) ** 2 * np.diagonal(self._K)
        # print(term2.shape, term5.shape)
        res = 2. * term1 + 2. * term2 - term3 - 2. * term4 #- term5
        return res

class MMD_comp(Function):
    '''
        F(S+s) - F(S) for MMD (Kim et al NIPS 2016)
        = AvgDiv + 2 * AvgCover
        = selects from VB

    '''
    def __init__(self, KB, KA, KAB, fB, fA, lambdaa = 0.2, diff = 1, **kwargs):
        super().__init__(len(KB))
        assert diff in {0, 1}
        self._fB = fB.detach().squeeze()
        self._fA = fA.detach().squeeze()
        # print(describe(self._fB.numpy()))
        self.pVB = F.softmax( self._fB, dim = 0 ).numpy()
        self.pVA = F.softmax( self._fA, dim = 0 ).numpy()
        # print("pB", describe(self.pVB))
        # print("pA", describe(self.pVA))
        
        # print(self.pV)
        self._KB = KB.numpy()
        self._KA = KA.numpy()
        self._KAB = KAB.numpy()
        assert self._KB.shape == ( len(self._fB), len(self._fB) )
        assert self._KA.shape == ( len(self._fA), len(self._fA) )
        assert self._KAB.shape == ( len(self._fA), len(self._fB) )
        logger.debug((self._KB.shape, self._KA.shape, self._KAB.shape, 
                self._fA.shape, self._fB.shape))

        self._rB = np.dot(self._KB, self.pVB)
        self._o1 = np.dot(self.pVB, self._rB)
        self._rA = np.dot(self._KAB.T, self.pVA)
        self._c1 = np.dot(self.pVA, np.dot(self._KA, self.pVA))
        self._CB = np.max(self._fB.numpy())
        self._CA = np.max(self._fA.numpy())
        assert len(self._rA) == len(self._rB)
        self.lambdaa = lambdaa
        assert self.lambdaa >= 0.0 and self.lambdaa < 0.9
        self.diff = diff

    def add2S(self, s):
        return super().add2S(s)

    def F(self, S = None, **kwargs):
        S_ = self._S if S is None else copy(S)
        lenS = (self._nS if S is None else np.sum(S))
        S_ = np.where(S_ == 1)[0]
        mmd = self._o1 - self.lambdaa * self._c1
        if lenS > 0:
            # print(S, self._K.shape )
            qS = F.softmax(self._f[S_].squeeze(), dim = 0).numpy()
            K_SS = self._K[S_, :][:, S_].squeeze()
            mmd -= 2.0 * np.dot(self._rB[S_], qS)
            mmd += 2.0 * self.lambdaa * np.dot(self._rA[S_], qS)
            mmd += ( 1 - self.lambdaa * self.diff ) * np.dot(qS, np.dot( K_SS, qS ))
        return -mmd
    
    def marginal_gain(self, S = None, **kwargs):
        S_ = self._S if S is None else copy(S)
        lenS = (self._nS if S is None else np.sum(S))
        S_ = np.where(S_ == 1)[0]
        e_fSB, e_fSA = 0.0, 0.0
        term1B, term1A, term3B, term4B = 0.0, 0.0, 0.0, 0.0
        e_fskB = np.exp(self._fB.numpy() - self._CB)
        
        if lenS > 0:
            qS = F.softmax(self._fB[S_].squeeze(), dim = 0).numpy()
            
            e_fSB = np.sum( np.exp(self._fB[S].numpy() - self._CB) )

            term1B = np.dot(self._rB[S_], qS) * (- e_fskB / (e_fskB + e_fSB) )
            term1A = np.dot(self._rA[S_], qS) * (- e_fskB / (e_fskB + e_fSB) )
            
            term3B = np.dot(qS, np.dot(self._KB[S_, :][:, S_].squeeze(), qS)) * \
                            (e_fSB ** 2 - (e_fSB + e_fskB) ** 2) / (e_fSB + e_fskB ) ** 2

            tmp = (e_fskB / (e_fSB + e_fskB)) * 1./ (1. + e_fskB / e_fSB).squeeze()
            term4B = np.dot(self._KB[:, S_], qS).squeeze() * tmp 
            # print(term1.shape, term3.shape, term4.shape)
        # print(type(self._r), type(e_fsk))
        term2B = self._rB * e_fskB / ( e_fskB + e_fSB )
        term2A = self._rA * e_fskB / ( e_fskB + e_fSB )
        term5B = ( e_fskB / (e_fSB + e_fskB) ) ** 2 * np.diagonal(self._KB)
        # print(term2.shape, term5.shape)
        res = 2. * ( term1B + term2B  ) 
        res -= (1.0 - self.lambdaa * self.diff) * ( term3B + 2. * term4B ) #+ term5B )
        res -= 2 * self.lambdaa * (term1A + term2A)
        return res

# https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=5, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def __repr__(self):
        return "state:{:.6g}, #bad_epochs:{}".format(0.0 if self.best is None else self.best.item(), self.num_bad_epochs)

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
                