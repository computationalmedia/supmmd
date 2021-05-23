import numpy as np
from profilehooks import profile
from commons.utils import get_logger
from copy import deepcopy
__logger_greedy = get_logger("greedy")

def greedy(F, candidates = None, budget = 5, costs = None, 
                r = 0.0, lb_cost = None, ub_cost = None, **kwargs):
    '''
        greedily maximize with cardinality or budgeted constraints
        - F: subclass submodular.Function
        - candidates: search candidates {0,1}^N vector, o indicate not candidate
                        by default it will be 1^N
        - budget: budget for subset to be searched (or cardinality)
        - costs: costs of each item R_+^N
        - r: hyperparameter for budgeted maximisation ( see [1] )
        - lb_cost: lower bound on cost, don't select items with cost < cost_lb 
        - ub_cost: upper bound on cost, don't select items with cost > cost_ub 

        [1] Lin, Hui, and Jeff Bilmes. "Multi-document summarization via budgeted maximization of submodular functions." Human Language Technologies: The 2010 Annual Conference of the North American Chapter of the Association for Computational Linguistics. 2010.
    '''
    cost = 0.0
    if costs is not None:
        assert np.all(costs > 0)
        _costs_provided = True
    else:
        _costs_provided = False
        costs = np.ones(F.universe_size)
    _cost_check = (lb_cost is not None) and (ub_cost is not None)
    if _cost_check:
        assert lb_cost <= ub_cost
        assert _costs_provided

    if candidates is None:
        candidates = np.ones(F.universe_size)
    
    assert len(costs) == len(candidates) == F.universe_size, (len(costs), len(candidates), F.universe_size)
    __logger_greedy.debug( (len(costs), len(candidates), F.universe_size) )
    S = [] ## F.S works, but this preserves the order
    keys = kwargs.get("keys")
    sel_keys = set()
    # __logger_greedy.info("keys" + str(keys))
    if keys is not None:
        assert len(keys) == len(costs)
        

    while budget > 0 and np.sum(candidates) > 0:
        marginal_gain = deepcopy(F.marginal_gain(**kwargs))
        assert len(marginal_gain) == len(costs), (len(marginal_gain), len(costs))
        marginal_gain -= 1e9 * (1-candidates)
        ix = np.argmax(marginal_gain / np.power( costs, r ))
        __logger_greedy.debug(( len(costs), marginal_gain.shape, ix ))
        # if marginal_gain[ix] < 0: continue
        # item = candidates[ix]
        sel = False
        cost_valid = not _cost_check or ( not _costs_provided or (costs[ix] >= lb_cost and costs[ix] <= ub_cost))
        if keys is not None:
            cost_valid &= ( keys[ix] not in sel_keys)
        if cost_valid: #and (budget - costs[ix] >= 0):
            #     continue
            sel = True
            budget -= costs[ix]
            cost += costs[ix]
            F.add2S(ix)
            S.append(ix)
            if keys is not None:
                sel_keys.add(keys[ix])

        candidates[ix] = 0
        log = "%s(s=%d, |S|=%d) = %+.4f, |s|=%d; cost=%d, %s"%( 
            F.__class__.__name__, ix, len(F),  marginal_gain[ix], 
            np.sum(candidates), cost, "+" if sel else "-" )
        
        __logger_greedy.debug(log)
        if kwargs.get("verbose") == True:
            print(log)
    ## verify if S is consistent with F.S
    assert set(np.where(F.S)[0].tolist()) == set(S)
    log = "selected {} prototypes, cost={:.3g}".format(len(F), cost)
    __logger_greedy.debug(log)
    if kwargs.get("verbose") == True:
        print(log)
    return S, cost

def cos(a_vals, b_vals):
    words  = list(a_vals.keys() | b_vals.keys())
    a_vect = [a_vals.get(word, 0) for word in words]        
    b_vect = [b_vals.get(word, 0) for word in words]

    # find cosine
    len_a  = sum(av*av for av in a_vect) ** 0.5 + 1e-6             
    len_b  = sum(bv*bv for bv in b_vect) ** 0.5 + 1e-6            
    dot    = sum(av*bv for av,bv in zip(a_vect, b_vect))  
    cosine = dot / (len_a * len_b)
    return cosine    

def select_from_scores(sents, scores, lengths, thresh = 0.5, 
            budget = 120, min_length = 8, max_length = 55):
    sorted_idx = np.argsort(1-scores)
    assert len(sents) == len(scores) == len(lengths)

    summ_length = 0
    idxs = [sorted_idx[0]]
    summ = sents[sorted_idx[0]]

    for idx in sorted_idx[1:]:
        if summ_length > budget:
            return idxs, summ_length
        if summ_length + lengths[idx] > budget or \
                lengths[idx] < min_length or lengths[idx] > max_length:
            continue
        
        score = cos(sents[idx], summ)
        if score < thresh :
            idxs.append(idx)
            summ += sents[idx]
            summ_length += lengths[idx]
    
    return idxs, summ_length