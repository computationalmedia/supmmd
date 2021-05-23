import json, pickle
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer as TfIdf
from commons.tokenize_utils import word_tokenize1
import sys
import numpy as np
from commons.utils import get_logger
logger = get_logger("idf")

def lemma_tok(w):
    return word_tokenize1(w, stemming = False, stop_words = True, lemmatize = True)

def tok2(w):
    return word_tokenize1(w, stemming = True, stop_words = False)

units = sys.argv[1]
assert units in ["stem", "lemma", "stem2"]

paths_ = [
        # ("duc", ["../oracle/duc03.sents", "../oracle/duc04.sents" ]), 
        # ("tac", ["../oracle/tac08.sents", "../oracle/tac09.sents" ]),
        ("x", ["../oracle/duc03.sents", "../oracle/duc04.sents", "../oracle/tac08.sents", "../oracle/tac09.sents" ]) 
]
sents = []
logger.info("starting")
toks = {"stem": word_tokenize1, "lemma": lemma_tok, "stem2": word_tokenize1}

for dataset, paths in paths_:
    sents = []
    for path in paths:
        logger.info("reading file: " + path)
        with open(path, "r") as fp:
            sents.extend(fp.readlines())
    
    vec = TfIdf(min_df = 2, max_df = 0.1, analyzer = 'word', 
                    lowercase = False, tokenizer = toks[units], 
                    preprocessor = None, ngram_range = (1, 2 if units == "stem2" else 1),
                    smooth_idf=True, norm = None
    )

    N = len(sents)
    logger.info("#sents:{}, fitting tfidf".format(len(sents)))
    M = vec.fit_transform(sents)
    logger.info("fitted, vocab size: %d"%len(vec.vocabulary_))
    logger.info("saving to file")

    with open("stop-%sx.txt"%units, "w") as fp:
    	logger.info("#stopwords:{}".format(len(vec.stop_words_)))
    	fp.write("\n".join(vec.stop_words_))


    tfisf = np.array(M.sum(axis=0)).squeeze()
    vocab = sorted(vec.vocabulary_.items() ,  key=lambda x: x[1])
    vocab = [k for k,v in vocab]
    idf = vec.idf_
    tf = tfisf/idf
    df = (1. + N) / np.exp(idf -1) -1

    with open("./vocab-%s-%s.csv"%(dataset, units), "w") as fp:
        fp.write("term,tf,df,idf\n")
        i = 0
        for ix in np.argsort(df):
            fp.write('"{}","{:g}","{:g}","{:.6g}"\n'.format(vocab[ix], tf[ix], df[ix], idf[ix]))
            i += 1
            if i % 100 == 0:
                fp.flush()
    logger.info("done")