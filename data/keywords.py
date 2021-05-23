from commons.tokenize_utils import word_tokenize1
from commons.gow import core_dec, terms_to_graph
import igraph
import pandas as pd
from itertools import chain, product
import numpy as np
import json
window = 4

for dataset in ["tac08_icsi", "tac09_icsi"]:
    path_sents = "../data/{}-sents-icsi.csv".format(dataset)
    sents = pd.read_csv(path_sents).fillna({'set':''})
    sents_df = sents[["sent", "num_words", "group", "set"]]
    sents_df = sents_df.query('num_words >= 8 & num_words <= 55').reset_index()
    sets = set(sents_df.set)
    topics = sorted(product( set(sents_df['group'].values), sets ))
    fo = open("../data/KW{}_{}.json".format(window, dataset), 'w')
    res = dict()
    for group, set_ in topics:
        idxs = np.where(np.logical_and(
                sents_df['group'].values == group, sents_df['set'] == set_
            ))[0]

        G = igraph.Graph(directed=True)
        
        sents = []
        for s in sents_df['sent'].iloc[idxs].values :
            tokens = word_tokenize1(s)
            tokens = [t for t in tokens if t.isalpha()]
            sents.append(tokens)
        G = terms_to_graph(sents, window)
        scores = core_dec(G, True)
        keywords_dict = sorted(scores.items(), key = lambda kv: -kv[1])
        print(dataset, group, set_, keywords_dict[:20])
        group_keywords = [k for k,v in keywords_dict ]
        res[ (group + ('' if len(sets) <= 1 else "_" + set_ ) ) ] = keywords_dict
    json.dump(res, fo)

        