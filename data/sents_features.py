from commons.tokenize_utils import word_tokenize1
from sklearn.feature_extraction.text import CountVectorizer as BoW
from lexrank.algorithms.power_method import stationary_distribution
import pandas as pd
import sys, json
from itertools import chain, product
import numpy as np
from commons.utils import get_logger
from sklearn.metrics.pairwise import pairwise_kernels
from nltk.tokenize import word_tokenize
import nltk
import re

logger = get_logger("data")

target_name = "y_hm_0.4"

dataset = sys.argv[1]
# assert dataset in {"duc03", "duc04", "tac08", "tac09" }
df = pd.read_csv("./{}-sents-punkt.csv".format(dataset)).fillna({'set':''})

## for dataset with unknown coref variable, oracle should not be sent with unresolved personal pronoun
if "up" in dataset:
    assert ((df["UnresPPR"].values > 0) * df["y_hm_0.4"].values ).sum() == 0

vocab_path = "../scripts/vocab-stem.csv"
logger.info("loading vocab: "+ vocab_path)
vocab_df = pd.read_csv(vocab_path )
vocab = vocab_df['term'].values
idf = vocab_df['idf'].values
idf = (idf / np.max(idf))

bow_all = BoW( analyzer = 'word', lowercase = False, tokenizer = word_tokenize1, 
        preprocessor = None, ngram_range = (1, 1),
        vocabulary = dict(zip(vocab, np.arange(len(vocab))))
)

query_titles = {}
query_narratives = {}
query_feats = False
try:
    with open(re.sub(r'(_up)', '', "{}_icsi.json".format(dataset)), "r") as fp:
        for line in fp:
            row = json.loads(line)
            group = row["group"]
            set_ = row["set"]
            query_titles[(group, set_)] = row["query.title"]
            query_narratives[(group, set_)] = row["query.narr"]

except Exception as ex:
    logger.error("error loading query " + str(ex))
# bow_all.fit(query_titles.values())
# bow_all.fit(query_narratives.values())

def sim_query(title, narr, txt):
    title_toks = word_tokenize1(title)
    narr_toks = word_tokenize1(narr)
    txt_toks = word_tokenize1(txt)

    query = set(narr_toks) | set(title_toks)

    score1 =  len( set(title_toks) & set(txt_toks) ) / len(set(title_toks))
    score2 =  len( set(narr_toks) & set(txt_toks) ) / len(set(narr_toks))
    score3 =  len( query & set(txt_toks) ) / len(query)
    return score1, score2, score3

sets = set(df['set'].values)
topics = sorted(product( set(df['group'].values), sets ))
if len(query_titles.keys()) == len(query_narratives.keys()) == len(topics):
    query_feats = True
else:
    logger.error((len(query_titles.keys()), len(query_narratives.keys()), len(topics)))

def _kernel(X, kernel, gamma = 1.0):
    if kernel == "exc":
        return np.exp(gamma * pairwise_kernels(X, metric = "cosine")) 
    elif kernel == "exl":
        return np.exp(gamma * pairwise_kernels(X, metric = "linear"))
    elif kernel == "inp":
        temp = 1.0 - pairwise_kernels(X, metric = "cosine")
        return temp ** -gamma
    elif kernel == "rbf":
        return pairwise_kernels(X, metric = "rbf", gamma = gamma)
    elif kernel == "cos":
        return pairwise_kernels(X, metric = "cosine")
sfs = []


def count_PRP(txt):
    tokens = word_tokenize(txt)
    tagged = list(nltk.pos_tag(txt))
    pronouns = len([tag for word, tag in tagged if tag == "PRP"])
    return pronouns

for group, set_ in topics:
    idxs = np.where(np.logical_and(
        df['group'].values == group, df['set'].values == set_
    ))[0]
    subset = df.iloc[idxs].reset_index()
    logger.info("topic:{}{}, |V|:{}".format(group, set_, len(subset)))
    S = np.where(subset[target_name].values == 1)[0]
    assert subset[target_name].values.sum() == len(S)
    assert set(subset['group'].values) == {group}, set(subset['group'].values)
    assert set(subset['set'].values) == {set_}, set(subset['set'].values)

    X = bow_all.transform(subset['sent'].values)

    sf = np.array( (X > 0).sum(axis=0)).squeeze()
    assert len(sf) == X.shape[1]
    non_zero_feats = np.where( sf > 0 )[0]
    sf = sf[non_zero_feats].squeeze()
    isf = np.log(len(idxs)) - np.log(sf)
    isf = isf / np.max(isf)
    assert len(isf) == len(non_zero_feats)
    
    X = np.array(X[:, non_zero_feats].todense()).squeeze()
    logger.info("topic:{}{}, BoW.shape:{}, #summ_sents:{}".format(
        group, set_, X.shape, len(S)
    ))

    lengths = (X > 0).sum(axis=1).squeeze()
    tf_avg = X.sum(axis=1)/lengths
    isf_avg = (X > 0).dot(isf) / lengths
    
    assert len(tf_avg) == len(isf_avg) == len(lengths) == len(idxs)
    
    X_logtf = np.log(X + 1.0)
    X_tfisf = X * isf
    X_logtfisf = X_logtf * isf
    topic_ = X_tfisf.sum(axis = 0)
    sim_ = X_tfisf.dot(topic_) / (np.linalg.norm(X_tfisf) * np.linalg.norm(topic_))
    assert X_logtf.shape == X_tfisf.shape == X.shape == X_logtfisf.shape == (len(idxs), len(non_zero_feats))
    
    # K = _kernel(X_tfisf, "exc", 2.0)
    A = _kernel(X_tfisf, "cos")
    np.fill_diagonal(A, 0.0)
    d = A.sum(axis=1, keepdims=True)
    lex_rank_scores = stationary_distribution(A/d, normalized = False)
    logger.info(("lex_rank_topic: ",  round(lex_rank_scores.sum(),0), round(lex_rank_scores.min(), 4), round(lex_rank_scores.max(), 4) ))
    tfisf_sum = X_tfisf.sum(axis = 1)
    logtf_sum = X_logtf.sum(axis = 1)
    logtfisf_sum = X_logtfisf.sum(axis = 1)
    assert len(tfisf_sum) == len(idxs) == len(logtf_sum) == len(logtfisf_sum), \
        (len(tfisf_sum), len(idxs), len(logtf_sum), len(logtfisf_sum))

    isf_all = np.array(idf[non_zero_feats]).squeeze()
    isf_all_avg = (X > 0).dot(isf_all) / lengths

    assert len(tf_avg) == len(isf_all_avg) == len(lengths) == len(idxs)
    
    X_tfisf_all = X * isf_all
    X_logtfisf_all = X_logtf * isf_all
    assert X_tfisf_all.shape == X_logtfisf_all.shape == (len(idxs), len(non_zero_feats))
    
    # K_all = _kernel(X_tfisf_all, "exc", 2.0)
    A_all = _kernel(X_tfisf_all, "cos")
    np.fill_diagonal(A_all, 0.0)
    d = A_all.sum(axis=1, keepdims=True)
    d[d < 1e-6] = 1e-6
    logger.info( ("#!0: ", (d >0).sum(), len(d)))
    lex_rank_scores_all = stationary_distribution(A_all/d, normalized = False)
    logger.info(("lex_rank_all: ", round(lex_rank_scores_all.sum(), 0), round(lex_rank_scores_all.min(), 4), round(lex_rank_scores_all.max(), 4) ))
    tfisf_sum_all = X_tfisf_all.sum(axis = 1)
    logtfisf_sum_all = X_logtfisf_all.sum(axis = 1)
    
    topic_all = X_tfisf_all.sum(axis = 0)
    sim_all = X_tfisf_all.dot(topic_all) / (np.linalg.norm(X_tfisf_all) * np.linalg.norm(topic_all))
    assert len(tfisf_sum_all) == len(idxs) == len(logtfisf_sum_all), (len(tfisf_sum_all), len(idxs), len(logtfisf_sum_all))
    pronouns = [ count_PRP(s) for s in subset['sent'].values ]

    surf_feats = np.vstack((
                subset.group.values,
                subset.set.values,
                subset.doc_id.values,
                subset.sent_id.values,
                subset.position.values,
                subset.doc_sents.values,
                lengths,
                subset.num_words.values,
                tf_avg,
                isf_avg,
                tfisf_sum,
                logtf_sum,
                logtfisf_sum,
                isf_all_avg,
                lex_rank_scores,
                tfisf_sum_all,
                logtfisf_sum_all,
                lex_rank_scores_all,
                pronouns,
                sim_,
                sim_all
            )).T
    sf_subset = pd.DataFrame(surf_feats, columns = [
                        "group", "set", "doc_id", "sent_id", "position", "doc_sents",
                        "vocab_words", "num_words", "tf_avg", "isf_avg", 
                        "tfisf_sum", "logtf_sum", "logtfisf_sum", "isf_all_avg", 
                        "lexrank", "tfisf_sum_all", "logtfisf_sum_all", "lexrank_all" , 
                        "pronouns", "sim", "sim_all"
    ])
    
    if query_feats:
        q_scores = [ sim_query(query_titles[(group, set_)], query_narratives[(group, set_)], s )\
                             for s in subset['sent'].values ]
        q_scores = np.array(q_scores)
        assert q_scores.shape == ( len(idxs), 3 )
        sf_subset = sf_subset.assign(title_sim = q_scores[:, 0], narr_sim = q_scores[:, 1], query_sim = q_scores[:, 2])
        logger.info("query: " + str(q_scores.mean(axis=0)) )
        logger.debug(sf_subset.columns)

    sf_subset["nouns"] = subset.nouns.values
    sf_subset["prpns"] = subset.prpns.values
    sf_subset["UnresPPR"] = subset.UnresPPR.values

    sfs.append(sf_subset)  
    assert np.all(surf_feats[:, 3] == subset['sent_id'].values)
    assert np.all(surf_feats[:, 0] == subset['group'].values)
    assert np.all(surf_feats[:, 1] == subset['set'].values)
    assert np.all(surf_feats[:, 2] == subset['doc_id'].values)
sf_df = pd.concat(sfs)

col_list = ["tf_avg", "isf_avg", "tfisf_sum", "logtf_sum", "logtfisf_sum", "isf_all_avg", 
                "lexrank", "tfisf_sum_all", "logtfisf_sum_all", "lexrank_all",
                "pronouns", "sim", "sim_all"
]

if query_feats:
    col_list += [ "title_sim", "narr_sim", "query_sim" ]

col_list += ["nouns", "prpns", "UnresPPR"]

sf_df[col_list] = sf_df[col_list].apply(pd.to_numeric, downcast='float')

sf_df.to_csv("./{}-surf-feats.csv".format(dataset), mode = 'w', 
    index = False, line_terminator = '\n',
    encoding = 'utf-8', float_format = '%.4g'
)
logger.info("done")