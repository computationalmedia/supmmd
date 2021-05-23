import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import sys, json
from itertools import chain, product
import pandas as pd
from commons.utils import get_logger
from commons.tokenize_utils import word_tokenize1
from sklearn.feature_extraction.text import CountVectorizer as BoW
from sklearn.feature_extraction.text import TfidfVectorizer as TfIsf
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import OneHotEncoder as OHE
from lexrank.algorithms.power_method import stationary_distribution
from copy import copy
# from sklearn.metrics import pairwise_distances
from commons.gow import core_dec, terms_to_graph
import pickle
from sup_mmd.llr import llr_compare

logger = get_logger("data")
ID_COLS = ["group", "set", "doc_id", "sent_id"]
BOOST_A = 1.0
BOOST_B = 2.0

def minmax_scaler(x):
    return (x - np.min(x, axis = 0)) / ( np.max(x, axis = 0) - np.min(x, axis = 0) )

def standard_scaler(x):
    return (x - np.mean(x, axis = 0)) / ( np.std(x, axis = 0) )

def sim_query(title, narr, txt):
    title_toks = word_tokenize1(title)
    narr_toks = word_tokenize1(narr)
    txt_toks = word_tokenize1(txt)
    query = set(narr_toks) | set(title_toks)
    # score1 =  len( set(title_toks) & set(txt_toks) ) / len(set(title_toks))
    # score2 =  len( set(narr_toks) & set(txt_toks) ) / len(set(narr_toks))
    score3 =  len( query & set(txt_toks) ) / len(query)
    return score3

tok4bigrams = lambda w: word_tokenize1(w, stemming = True, stop_words = False) ## no stopwords in bigrams
# sent,num_words,position,group,set,doc_id,sent_id,doc_sents,
# R1.P,R1.R,R1.F,R2.P,R2.R,R2.F,UnresPPR,nouns,prpns,
# y_hm_0.0,y_R2_0.0,y_hm_0.1,y_R2_0.1,y_hm_0.2,y_R2_0.2,y_hm_0.3,y_R2_0.3,y_hm_0.4,y_R2_0.4,y_hm_0.5,y_R2_0.5

class MMD_Dataset(Dataset):
    def __init__(self, dataset, compress = False, target_name = 'y_hm_0.4'):
        self.target_name = target_name
        self._name = "{}_{}".format(dataset, target_name)
        self.dataset = dataset
        self.compress = compress
        ## load dataset
        self._logger = get_logger("SummDataset: {}".format(self._name))
        assert self.dataset in {"duc03", "duc04", "tac08", "tac09"}
        
        ## load sentences
        if self.dataset in {"duc03", "duc04"}:
            path_sents = "../data/{}{}-sents-punkt.csv".format(self.dataset, "-expanded" if self.compress else "")
        elif self.dataset in {"tac08", "tac09"}:
            path_sents = "../data/{}_icsi2{}-sents.csv".format(self.dataset, "-expanded" if self.compress else "")

        self._logger.info("loading data from " + path_sents )
        self.df = pd.read_csv(path_sents).fillna({'set':'A'})
        assert self.target_name in self.df.columns.values
        cols = ["sent", "group", "set", "doc_id", "sent_id", "doc_sents",
                                "R1.R", "R1.P", "R2.R", "R2.P", "UnresPPR",
                                "nouns", "prpns", "num_words", "position", self.target_name]
        self.sets = set(self.df["set"].values)
        if self.dataset in {"duc03", "duc04"}:
            assert self.sets == {'A'}
        else:
            assert self.sets == {'A', 'B'}
        self.comp_names = []

        if self.dataset in {"tac08", "tac09"}:
            query_titles, query_narratives = self._load_queries_tac()
            self._logger.info("computing query similarity")
            self.df["qsim"] = self.df.apply (lambda row: sim_query(
                query_titles[(row["group"], row["set"])], 
                query_narratives[(row["group"], row["set"])],
                row["sent"]
            ), axis=1)
            # self.df["qsim"] = 0.1
            self._logger.info("query similarity computed")
            cols += [ "par_start", "qsim"]
        ## filter sents
        self.df = self.df[cols].query('num_words >= 8 & num_words <= 55').rename( columns = {target_name : 'target'}  ).reset_index()
        self.df.set_index(ID_COLS, inplace=False)

        self._logger.info("loaded txt from {}, #rows:{}".format(path_sents, len(self.df) ))
        self._logger.info(list(self.df.columns))
        
        ## load surface features
        self._logger.info("loading surface features")
        self._init_surface_feats() ## num_words, position info
        
        ## find indices of each group
        self.groups = sorted( set(self.df['group'].values) )
        self._load_indices()

        ## load kernels and compute lexrank scores
        self._logger.info("loading kernels")
        if self.dataset in {"duc03", "duc04"}:
            self._load_kernels_duc()
        else:
            self._load_kernels_tac()

        ## validation
        for g, K in self.kernels.items():
            assert K.shape[2] == self.num_kernels, ( K.shape, self.num_kernels )  

        self._logger.info("computing lexrank")
        self._compute_lexrank()

        self._logger.info("loading keywords")
        self._load_keywords()

        self.surface_feats = torch.DoubleTensor( self.surface_feats )

        assert self.surface_feats.shape[1] == len(self.surf_names), (self.surface_feats.shape[1], self.surf_names)
        self._logger.info("Finished loading dataset:{}, #groups:{}, #sents:{}, sf:{}, num_kernels:{}".format(
            self.dataset, len(self), len(self.df), self.num_surf_feats, self.num_kernels ))
        self._logger.info("Dataset summary: {}".format(str(self)))
        self.augment = False ## todo

    @property
    def num_surf_feats(self):
        return len(self.surf_names)

    @property
    def num_kernels(self):
        return len(self.kernel_names)

    def _compute_lexrank(self):
        lex_rank_scores_K = np.ones(len(self.df)) 

        for group, set_ in product(self.groups, self.sets):
            idxs = self.group_idxs[ (group, set_) ]
            Kt2 = self.kernels[(group, set_)][:, :, 1].numpy()

            K2 = copy(Kt2)
            np.fill_diagonal(K2, 0.0)
            d = K2.sum(axis=1, keepdims=True)
            lex_rank_scores_K[idxs] = stationary_distribution(K2/K2.sum(axis=1, keepdims=True), normalized = False)

        self.surface_feats = np.c_[ self.surface_feats, lex_rank_scores_K - 1.0 ]
        self.surf_names += ["lexrank"]

    def _load_queries_tac(self):
        query_title, query_narr = dict(), dict()
        with open("../data/{}_icsi.json".format(self.dataset), "r") as fp:
            for line in fp:
                row = json.loads(line)
                group = row["group"]
                set_ = row["set"]
                query_title[(group, set_)] = row["query.title"]
                query_narr[(group, set_)] = row["query.narr"]
        return query_title, query_narr

    def _first_sents_ents_idicator(self, X, pos):
        ixs = np.where(pos == 0)[0]
        res = (X[ixs, :].sum(axis = 0).squeeze() > 0)
        return res

    def __vectorize( self, vec, isf, idxs ):
        X = vec.transform( self.df['sent'].iloc[idxs].values )
        if isf is None:
            isf = np.ones(X.shape[1])
        sf = np.array( (X > 0).sum(axis=0)).squeeze()
        assert len(sf) == X.shape[1]
        non_zero_feats = np.where( sf > 0 )[0]
        X = np.array(X[:, non_zero_feats].todense()).squeeze()
        X = X * isf[non_zero_feats]
        return X

    ## load kernels, add tfisf scores to surface features
    def _load_kernels_duc(self):
        isf1, bow1 = self._load_vocab(1) ## unigrams
        isf2, bow2 = self._load_vocab(2) ## bigrams
        ## loading entities
        Xe1, _ = self._load_ents_and_vectorize() ## entities
        tfisf = np.zeros(len(self.df)) 
        btfisf = np.zeros(len(self.df)) 
        lex_rank_scores_K = np.ones(len(self.df)) 
        self.kernels = dict()
        
        for group in self.groups:
            idxs = self.group_idxs[(group, 'A')]
            Xt1 = self.__vectorize(bow1, isf1, idxs)
            Xt2 = self.__vectorize(bow2, isf2, idxs)

            Kt1 = pairwise_kernels(Xt1, metric = "cosine") #self._kernel(X)
            Kt2 = pairwise_kernels(Xt2, metric = "cosine")
            Ke1 = pairwise_kernels( Xe1[idxs, :].todense(), metric = "cosine")
            K = np.dstack((Kt1, Kt2, Ke1))
            self.kernels[(group, 'A')] = torch.DoubleTensor(K)
            self._logger.info("topic:{}{}, BoW.shape:{}, K.shape:{}".format(
                group, 'A', (Xt1.shape, Xt2.shape),  K.shape
            ))
            # Xt1 = Xt1 + 1.0 * Xt1 * self._first_sents_ents_idicator(X, self.df['pos'].iloc[idxs].values )
            tfisf[idxs] = 0.25 * (Xt1.sum(axis = 1).squeeze() - 8.0)
            res = self._first_sents_ents_idicator(Xt1, self.df['position'].iloc[idxs].values )
            Xt1 = Xt1 + BOOST_A * Xt1 * res
            self._logger.info("group:{}, #first tokens fraction: {:.4g}" .format ( 
                group, res.sum() / len(res)))
            btfisf[idxs] = ( Xt1.sum(axis = 1).squeeze() )
        self.kernel_names = ["unigrams", "bigrams", "entities"]
        ## tfisf scores to surface features
        self.surface_feats = np.c_[ self.surface_feats, tfisf, standard_scaler(btfisf) ]
        self.surf_names += ["tfisf", "btfisf"]

    ## doesn't work
    def _srllr_topK(self, XA, XB, K = 20):
         #### comp feature (top 20 Llr keywords)
        # self._logger.info(Xe1A.sum(axis=0))
        assert XA.shape[1] == XB.shape[1]
        counterA = {i:v for i,v in enumerate(XA.sum(axis=0)) if v > 0}
        counterB = {i:v for i,v in enumerate(XB.sum(axis=0)) if v > 0}
        srllr = llr_compare(counterB, counterA)
        comp_ents_idxs = [ i for i, v in sorted(srllr.items(), key=lambda x:-x[1])[:K] ]
        return standard_scaler(XB[:, comp_ents_idxs].sum(axis=1)), comp_ents_idxs

    def _novelity_factor(self, bow, idxsA, idxsB, isf = None):
        X = bow.transform( self.df['sent'].iloc[idxsA.tolist() + idxsB.tolist()].values )
        sf = np.array( (X > 0).sum(axis=0)).squeeze() > 0
        non_zero_feats = np.where( sf > 0 )[0]
        X = np.array(X[:, non_zero_feats].todense()).squeeze() > 0
        XA, XB = X[:len(idxsA), :], X[len(idxsA):, :]
        doc_idsA = OHE().fit_transform(self.df['doc_id'].iloc[idxsA].values.reshape(-1, 1))
        doc_idsB = OHE().fit_transform(self.df['doc_id'].iloc[idxsB].values.reshape(-1, 1))
        doc_idsA = np.array(doc_idsA.todense()).squeeze()
        doc_idsB = np.array(doc_idsB.todense()).squeeze()
        
        if type(isf) == np.ndarray:
            isf = isf[non_zero_feats]
        else:
            isf = 1.0
        # print(doc_idsA.shape, doc_idsB.shape, XA.shape, XB.shape, type(doc_idsA))
        dA = ( XA.T.dot(doc_idsA) > 0 ).sum(axis = 1)
        dB = ( XB.T.dot(doc_idsB) > 0 ).sum(axis = 1)
        # print(dA.shape, dB.shape)
        assert dA.shape == dB.shape
        XA = XA * dA * isf / doc_idsA.shape[1]
        XB = XB * dB * isf / (dA + doc_idsB.shape[1])
        fA, fB = XA.sum(axis = 1), XB.sum(axis = 1)
        return  fA, fB

    ## load kernels, add tfisf scores to surface features
    def _load_kernels_tac(self):
        isf1, bow1 = self._load_vocab(1) ## unigrams
        isf2, bow2 = self._load_vocab(2) ## bigrams
        ## loading entities
        Xe1, bow_ents = self._load_ents_and_vectorize() ## entities
        ents_vocab_all = np.array(sorted(bow_ents.vocabulary_, key = bow_ents.vocabulary_.get))
        
        # print( bow_ents )
        tfisf = np.zeros(len(self.df)) 
        btfisf = np.zeros(len(self.df)) 
        lex_rank_scores_K = np.ones(len(self.df)) 
        self.kernels = dict()
        # comp_feats = np.zeros(len(self.df))
        nf = np.zeros(len(self.df))
        cA, cB = 0.0, 0.0
        for group in set(self.df['group']):
            idxsA = self.group_idxs[(group, 'A')]
            idxsB = self.group_idxs[(group, 'B')]

            ## unigrams
            X1t = self.__vectorize( bow1, isf1, idxsA.tolist() + idxsB.tolist() )
            Xt1A = X1t[: len(idxsA), :] 
            Xt1B = X1t[len(idxsA): , :]
            fA1, fB1 = self._novelity_factor( bow1, idxsA, idxsB, 1.0 )
            nf[idxsA] = standard_scaler(fA1) 
            nf[idxsB] = standard_scaler(fB1)
            _cA, _cB = np.corrcoef(fA1, self.df['R2.R'].iloc[idxsA].values)[0, 1], np.corrcoef(fB1, self.df['R2.R'].iloc[idxsB].values)[0, 1]
            cA += _cA
            cB += _cB

            ## bigrams
            X2t = self.__vectorize(bow2, isf2, idxsA.tolist() + idxsB.tolist())
            Xt2A = X2t[: len(idxsA), :] 
            Xt2B = X2t[len(idxsA): , :]
            # fA2, fB2 = self._novelity_factor( bow2, idxsA, idxsB ) ## unigrams is good enough

            ## entities
            Xe1_ = np.array( Xe1[idxsA.tolist() + idxsB.tolist(), :].todense())
            # self._logger.info(Xe1_.sum(axis = 1))
            ents_idxs = np.where( Xe1_.sum(axis = 0) > 0 )[0]
            Xe1_ = Xe1_[:, ents_idxs ]
            Xe1A = Xe1_[: len(idxsA), :] 
            Xe1B = Xe1_[len(idxsA): , :]
            ents_vocab = ents_vocab_all[ents_idxs]
            self._logger.info("{}: #ents:{}".format(group, len(ents_vocab)))
            
            ## llr top ents
            comp_featsB, comp_ents_idxs = self._srllr_topK(Xt1A, Xt1B, 20)
            # self._logger.info("{}: comp_ents(B):{}".format(group, ", ".join([ ents_vocab[i] for i in comp_ents_idxs ]) ))
            comp_scores = ( Xt1B[:, comp_ents_idxs] > 0).sum(axis=1)

            self._logger.info("{}: NF coerr {:.3g},{:.4g}, srllr:{:.4g}".format(
                group, _cA, _cB, np.corrcoef(comp_scores, self.df['R2.R'].iloc[idxsB].values)[0, 1]
            ))
            # comp_feats[idxsB] = standard_scaler(comp_scores)
            
            ## kernels of A
            Kt1A = pairwise_kernels(Xt1A, metric = "cosine")
            Kt2A = pairwise_kernels(Xt2A, metric = "cosine")
            Ke1A = pairwise_kernels(Xe1A, metric = "cosine")
            KA = np.dstack((Kt1A, Kt2A, Ke1A))
            
            ### kernels of B
            Kt1B = pairwise_kernels(Xt1B, metric = "cosine")
            Kt2B = pairwise_kernels(Xt2B, metric = "cosine")
            Ke1B = pairwise_kernels(Xe1B, metric = "cosine")
            KB = np.dstack((Kt1B, Kt2B, Ke1B))
            
            ### kernels of AB
            Kt1AB = pairwise_kernels(Xt1A, Xt1B, metric = "cosine")
            Kt2AB = pairwise_kernels(Xt2A, Xt2B, metric = "cosine")
            Ke1AB = pairwise_kernels(Xe1A, Xe1B, metric = "cosine")
            KAB = np.dstack((Kt1AB, Kt2AB, Ke1AB))
            
            self.kernels[(group, 'A')] = torch.DoubleTensor(KA) 
            self.kernels[(group, 'B')] = torch.DoubleTensor(KB) 
            self.kernels[(group, 'AB')] = torch.DoubleTensor(KAB)
            self._logger.info("topic:{}, BoW1.shape:{}, K.shape:{}".format(
                group, (X1t.shape, X2t.shape), ( KA.shape, KB.shape, KAB.shape )
            ))

            ## tfisf scores
            tfisf[idxsA] = 0.25 * ( Xt1A.sum(axis = 1).squeeze() - 8.0 )
            tfisf[idxsB] = 0.25 * ( Xt1B.sum(axis = 1).squeeze() - 8.0 )

            resA = self._first_sents_ents_idicator(Xt1A, self.df['position'].iloc[idxsA].values )
            resB = self._first_sents_ents_idicator(Xt1B, self.df['position'].iloc[idxsB].values )
            Xt1A = Xt1A + BOOST_A * Xt1A * resA
            Xt1B = Xt1B + BOOST_B * Xt1B * resB

            btfisf[idxsA] = (Xt1A.sum(axis = 1).squeeze() )
            btfisf[idxsB] = (Xt1B.sum(axis = 1).squeeze() )

            self._logger.info("group:{}, #first tokens fraction-A: {:.4g}, -B:{:.4g}" .format ( 
                group, resA.sum() / len(resA), resB.sum() / len(resB)))
        
        self._logger.info("NF {:.3g},{:.4g}".format( cA/len(self), cB/len(self) ))
        self.kernel_names = ["unigrams", "bigrams", "entities"]
        ## tfisf scores to surface features
        self.surface_feats = np.c_[ self.surface_feats, tfisf, standard_scaler(btfisf), nf ]
        self.surf_names += ["tfisf", "btfisf", "nf"]
        self.comp_names += [ "nf" ]


    def _load_indices(self):
        self.group_V_S = dict()
        self.group_idxs = dict()
        self.summ_idxs = dict()
        nouns = np.zeros(len(self.df)) 
        qsim = np.zeros(len(self.df)) 
        for group, set_ in product(self.groups, self.sets):
            idxs = np.where(np.logical_and(
                self.df['group'].values == group, self.df['set'] == set_
            ))[0]
            self.group_idxs[(group, set_)] = torch.LongTensor(idxs.tolist())
            
            S = np.where(self.df.target.values[idxs] == 1)[0]
            assert self.df.target.values[idxs].sum() == len(S)
            self.summ_idxs[(group, set_)] = torch.LongTensor(S.tolist()) 
            self.group_V_S[(group, set_)] = torch.LongTensor( 
                np.setdiff1d(np.arange(len(idxs)).squeeze(), S ).squeeze().tolist()            
            )
            assert set(self.df["group"].iloc[idxs].values) == {group}
            assert set(self.df["set"].iloc[idxs].values) == {set_}
            nouns[idxs] = minmax_scaler( self.df['nouns'].iloc[idxs].values + self.df['prpns'].iloc[idxs].values )
            if self.dataset in {"tac08", "tac09"}:
                qsim[idxs] = np.nan_to_num( minmax_scaler(self.df["qsim"].iloc[idxs].values ) )
        ## add nouns to surface features
        self.surface_feats = np.c_[self.surface_feats, nouns]
        self.surf_names += ["#nouns"]
        if self.dataset in {"tac08", "tac09"}:
            self.surface_feats = np.c_[self.surface_feats, qsim]
            self.surf_names += ["query_sim"]

    ## count top 50 keywords in 4 bins
    def _load_keywords(self):
                ## load keywords
        with open("../data/KW4_{}.json".format(self.dataset)) as fp:
            keywords_data = json.loads(fp.read())
        ohe_keywords = np.zeros((len(self.df), 4))

        for group, set_ in product(self.groups, self.sets):
            idxs = self.group_idxs[(group, set_)]
            keywords_dict = sorted( 
                keywords_data[group + ('' if len(self.sets) <= 1 else "_" + set_ )],
                key = lambda kv: -kv[1]
            )[:50]
            group_keywords = [k for k,v in keywords_dict ]
            self._logger.debug(keywords_dict)
            bow_kw = BoW( analyzer = 'word', lowercase = False, 
                tokenizer = word_tokenize1, 
                preprocessor = None, ngram_range = (1, 1),
                vocabulary = group_keywords
            )
            KW_mat = np.array(bow_kw.transform(self.df['sent'].iloc[idxs].values).todense()).squeeze()
            temp = np.c_[ KW_mat[:, :10].mean(axis = 1), KW_mat[:, 10:20].mean(axis = 1), 
                            KW_mat[:, 20:30].mean(axis = 1), KW_mat[:, 30:].mean(axis = 1) 
            ]
            
            ohe_keywords[idxs, :] = standard_scaler(temp)
        self.surface_feats = np.c_[self.surface_feats, ohe_keywords]
        self.surf_names += ["kw_01-10", "kw_11-20", "kw_21-30", "kw_31-50"]

    ## vectorize n_grams
    def _load_vocab(self, n_grams):
        vocab_path = "../scripts/vocab-stem{}.csv".format("" if n_grams == 1 else "2")
        vocab_df = pd.read_csv(vocab_path)
        vocab = vocab_df['term'].values
        isf_scores = vocab_df['idf'].values
        isf_scores = (isf_scores / np.max(isf_scores))
        bow_pre = BoW( analyzer = 'word', lowercase = False, 
            tokenizer = (word_tokenize1 if n_grams == 1 else tok4bigrams), 
            preprocessor = None, ngram_range = (1, n_grams),
            vocabulary = dict(zip(vocab, np.arange(len(vocab))))
        )
        assert len(vocab) == len(bow_pre.vocabulary)
        self._logger.info("loaded vocab from: " + vocab_path)
        return isf_scores, bow_pre

    ## vectorize bag of entities
    def _load_ents_and_vectorize(self):
        ents = dict()
        ents_file_path = "../data/entities_{}.json".format(self.dataset)
        self._logger.info( "loading entities from " + ents_file_path )
        with open(ents_file_path, "r") as ents_file:
            for line in ents_file:
                annotations = {row[1].lower(): row[0] for row in json.loads(line)["annotations"] }
                ents.update(annotations)

        ents_vec = TfIsf( analyzer = 'word', lowercase = False,
                    tokenizer = lambda x: x, preprocessor = None, 
                    max_df = 0.2, ngram_range = (1,1), min_df = 2,
                    norm = None
        )
        
        ents_sents = [ [v for k, v in ents.items() if k in s.lower() ] for s in self.df["sent"] ]
        X_ents = ents_vec.fit_transform(ents_sents)
        logger.warning("%d sents has 0 entities"%len(np.where(X_ents.sum(axis=1) == 0)[0]))
        return X_ents, ents_vec

    def _init_surface_feats(self):
        self.surf_names = [] ## save the names of surface features
        rel_pos = 1 - self.df['position'].values / self.df['doc_sents'].values ## relative position
        ## absolute position
        abs_pos = OHE().fit_transform( np.minimum(self.df['position'].values, 3).reshape(-1, 1) ).todense().squeeze()
        num_words = 0.1 * (self.df[ "num_words" ].values - 25.0) ## num words
        self.surface_feats = np.c_[ abs_pos, rel_pos, num_words ]
        self.surf_names += ["rel_pos", "pos1", "pos2", "pos3", "pos4+", "#words"]
        if self.dataset in {"tac08", "tac09"}:
            self.surface_feats = np.c_[self.surface_feats, self.df["par_start"].values]
            self.surf_names += ["par_start"]

    def get_subset_df(self, group, set_):
        idxs_topic = self.group_idxs[(group, set_)]
        subset = self.df.iloc[idxs_topic]
        assert set(subset['group'].values) == {group}
        assert set(subset['set'].values) == {set_}
        return subset

    def __len__(self):
        return len(self.groups)

    def cuda(self):
        self.summ_idxs = {k: v.cuda() for k,v in self.summ_idxs.items() }
        self.group_V_S = {k: v.cuda() for k,v in self.group_V_S.items() }
        self.surface_feats = self.surface_feats.cuda()
        self.kernels = {k: v.cuda() for k,v in self.kernels.items() }
        self.group_idxs = {k: v.cuda() for k,v in self.group_idxs.items() }
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx) or type(idx) == np.ndarray:
            # print(type(idx), idx)
            idx = idx.tolist()
            assert len(idx) == 1, "per topic processing only"
            idx = idx[0]

        group = self.groups[idx]
        if self.dataset in {"duc03", "duc04"}:    
            K = self.kernels[(group, 'A')]
            X = self.surface_feats[self.group_idxs[(group, 'A')], :]
            S = self.summ_idxs[(group, 'A')]
            V_S = self.group_V_S[(group, 'A')]
            return K, X, S, V_S
        else:
            VA = self.group_idxs[(group, 'A')]
            VB = self.group_idxs[(group, 'B')]
            KA = self.kernels[(group, 'A')]
            KB = self.kernels[(group, 'B')]
            KAB = self.kernels[(group, 'AB')]

            XA = self.surface_feats[VA, :]
            XB = self.surface_feats[VB, :]
            SA = self.summ_idxs[(group, 'A')]
            SB = self.summ_idxs[(group, 'B')]
            V_SA = self.group_V_S[(group, 'A')]
            V_SB = self.group_V_S[(group, 'B')]
            
            return KA, KB, KAB, XA, XB, SA, SB, V_SA, V_SB

    def __repr__(self):
        desc = "Dataset: {}{}\n  - #groups: {}, #sets: {}\n  " + \
                "- labels: {}\n  - #sents: {}, #summary sents: {}\n  "+ \
                "- surface features [{}]: {}\n  - kernels [{}]: {} "
        return desc.format(
            self.dataset.upper(),
            "[expanded by compression]" if self.compress else "",
            len(self.groups), len(self.sets),
            self.target_name,
            len(self.surface_feats) if len(self.sets) == 1 else "A={},B={}".format(
                sum(self.df['set'].values == "A"),
                sum(self.df['set'].values == "B")
            ), 
            self.df.target.values.sum(),
            self.num_surf_feats, ", ".join(self.surf_names),
            self.num_kernels, ", ".join(self.kernel_names)
        )

    @staticmethod
    def load(name, CACHE_ROOT = "./data/", compress = False):
        with open("{}/{}{}.pik".format(CACHE_ROOT, name, "c" if compress else ""), "rb") as fp:
            dataset = pickle.load(fp)
        logger.info("loaded from " + "{}/{}{}.pik".format(CACHE_ROOT, name, "c" if compress else ""))
        return dataset
        
    def save(self, CACHE_ROOT = "./data/"):
        with open("{}/{}{}.pik".format(CACHE_ROOT, self._name, "c" if self.compress else ""), "wb") as fp:
            pickle.dump(self, fp)
            logger.info("saved to " + "{}/{}{}.pik".format(CACHE_ROOT, self._name, "c" if self.compress else ""))

    def surf_idxs(self, keywords = False, boost_first = True, comp = False):
        idxs = []
        for ix, sf_name in enumerate(self.surf_names):
            if not keywords and ( "kw_" in sf_name ):
                continue
            if not comp and ( sf_name in self.comp_names ):
                continue
            if not boost_first and sf_name == "btfisf":
                continue
            idxs.append(ix)
        return torch.LongTensor(idxs) 

def main():
    name = sys.argv[1]
    compress = sys.argv[2] == "1"
    target = sys.argv[3]
    assert target in {"y_R2_0.0", "y_hm_0.4"}
    try:
        dataset = MMD_Dataset.load("{}_{}".format(name, target), "./data/", compress)
    except:
        dataset = MMD_Dataset(name, compress = compress, target_name = target)
        dataset.save("./data/")

    print(dataset)
    print(dataset.surface_feats.mean(axis=0))

    print(np.array(dataset.surf_names)[dataset.surf_idxs(keywords=False, boost_first= True, comp = False)])
    print(np.array(dataset.surf_names)[dataset.surf_idxs(keywords=True, boost_first= True, comp = True)])
    print(np.array(dataset.surf_names)[dataset.surf_idxs(keywords=False, boost_first= True, comp = True)])
    
    # from scipy.stats import describe, pearsonr
    # print(describe(dataset.surface_feats[:, -2]), 
    #             pearsonr( dataset.surface_feats[:, -2],  4.0 * ( dataset.surface_feats[:, -1] + 8 ) ))
    # print(dataset.surface_feats[:, -2])
if __name__ == '__main__':
    main()
