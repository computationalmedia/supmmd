## credits: parts adapted from https://github.com/nlpyang/PreSumm
import re, json
from itertools import chain
from collections import Counter
from functools import reduce
from copy import copy
import numpy as np
from rouge_score import rouge_scorer
import json, sys, time, pickle, traceback

from commons.utils import get_logger
import multiprocessing as mp
import pandas as pd
from commons.tokenize_utils import word_tokenize1
import spacy
nlp = spacy.load('en')
import neuralcoref
neuralcoref.add_to_pipe(nlp)

STEMMING = True
COMPRESS = True
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=STEMMING)

patterns = [
    re.compile(r'\(([a-z]+\s+)+[a-z]+\)$'),
    re.compile(r'^[^\.?!]+$'),
]
_logger = get_logger("oracle")

# sent_tokenize1 = nltk.tokenize.sent_tokenize
# def load_sent_tok():
try:
    path = '../commons/duc_tac_punkt.pik'
    _logger.debug("loading sent_tokenize from " + path)
    with open(path, "rb") as fp:
        # fp.seek(0)
        tok = pickle.load(fp)
        sent_tokenize1 = tok.tokenize
    _logger.info("loded sent_tokenize from " + path)
except Exception as ex:
    _logger.error("failed loading pretrained tokenizer")
    _logger.error(str(ex))
    traceback.print_stack()
    # traceback.print_tb(ex.__traceback__)
# load_sent_tok()


Rs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
pattern = re.compile( r'\n(he|she|ask|said|that|supported|program|and|prosecutor|in|or|queried|has|of|gave|as|over|who|according|than|wrote|there|they|an|a\s|at|then|it|his|her|the|will|shouted|write|is|but|moment|by|to|promised|sobbed|declared|came|cried|reported|was|were|being)')
word_tokenize = lambda w: word_tokenize1(w, stemming = True, stop_words = False)

def match(s):
    return reduce(lambda a, b: a or b,  [p.match(s) for p in patterns])

def _rouge_clean(s):
    return re.sub(r'[^a-zA-Z0-9 ]', '', s)

def hm_scorer(r1s, r2s, cost):
    return 2.0/cost *(r1s*r2s)/(r1s+r2s+1e-6)

def rouge1_scorer(r1s, r2s, cost):
    return r1s/cost

def rouge2_scorer(r1s, r2s, cost):
    return r2s/cost

def am_scorer(r1s, r2s, cost):
    return  0.5/cost * (r1s + r2s )

SCORERS = {
    'am': am_scorer,
    'hm': hm_scorer,
    'R1': rouge1_scorer, 
    'R2': rouge2_scorer
}

def listener(q, infile):
    oracles_path = infile.split("/")[-1].replace(".json", "-oracles-all.csv")
    sents_path = infile.replace(".json", "-sents-punkt-all.csv")
    
    _logger.info( "creating oracles file: " + oracles_path )
    _logger.info( "creating sents file: " + sents_path )

    fo = open(oracles_path, "w")
    fo.write("group,set,method,scorer,r,cost,len,rouge1_P,rouge1_R,rouge2_P,rouge2_R,idxs\n")

    fs = open(sents_path, "w")
    cols = ["sent", "num_words", "position", "group", "set", "doc_id", "sent_id", "seq_id", "doc_sents", \
                "R1.P", "R1.R", "R1.F", "R2.P", "R2.R", "R2.F", "UnresPPR", "nouns", "prpns"
    ]
    for r in Rs:
        for scorer, f in SCORERS.items():
            cols.append("y_{}_{}".format(scorer, r))
    fs.write(",".join(cols) + "\n")
    fs.close()

    while True:
        m = q.get()
        if m == 'kill':
            _logger.info("done processing all data")
            break
        oracles, sents = m
        
        for oracle in oracles:
            fo.write(oracle + "\n")
        fo.flush()
        df = pd.DataFrame(sents)
        df = df[cols]
        pd.DataFrame(df).to_csv(sents_path, mode = 'a', 
            header = False, index = False, line_terminator = '\n',
            encoding = 'utf-8', float_format = '%.4g'
        )
    fo.close()

class RougeGTCreator(object):
    def __init__(self, articles, abstracts):
        
        self.sentences = []
        self.num_words = []
        self.positions = []
        self.doc_sents =[]
        self.docids = []

        # self.par_starts = []
        self.unresolved_pronouns = []
        self.nouns = []
        self.prpns = []

        for article in articles:
            docid = article["docid"]
            # sents = list(nlp(article["body"]["text"].rstrip().replace('\n', ' ')).sents)
            sents = sent_tokenize1(article["body"]["text"].rstrip().replace('\n', ' ').replace(r'\s{2,}', ' '))
            sents = "\n".join(sents)
            sents = pattern.sub(r' \1', sents)
            sents = sents.split("\n")
            sents = [s for s in sents if len(s.strip()) > 1 ]
            sents = [s for s in sents if not match(s)]
            positions = np.arange(len(sents)) ##
            num_words = [ len([ w for w in word_tokenize(ss) ]) for ss in sents ]

            self.sentences.extend( sents )
            self.num_words.extend( num_words )
            self.positions.extend( positions ) ##
            self.doc_sents.extend( [len(sents)] * len(sents)  ) ##
            self.docids.extend( [ docid ] * len(sents)  )

            sents_ = [ nlp(s) for s in sents ]
            un_pron = [ len([t for t in s if t.pos_ == "PRON" and t._.in_coref]) for s in sents_ ]
            self.nouns.extend([ len([t for t in s if t.pos_ == "NOUN"]) for s in sents_ ])
            self.prpns.extend([ len([t for t in s if t.pos_ == "PROPN"]) for s in sents_ ])
            self.unresolved_pronouns.extend(un_pron)

        self.abstracts = abstracts
        assert len(self.sentences) > 0
        assert len(self.abstracts) > 0, "should provide multiple references"
        assert len(self.sentences) == len(self.num_words) == len(self.positions) == len(self.doc_sents) == len(self.docids)
        
        assert np.all(np.array(self.positions) <= np.array(self.doc_sents))

        self.sents_rouge1, self.sents_rouge2 = np.zeros((len(self), 3)), np.zeros((len(self), 3))

    def __len__(self):
        return len(self.sentences)
    
    def compute_rouge(self, idxs):
        pred = "\n".join([self.sentences[idx] for idx in idxs])
        scores = [scorer.score(abstr, pred) for abstr in self.abstracts]
        rouge1 = np.array([ [score['rouge1'].precision,score['rouge1'].recall, score['rouge1'].fmeasure] for score in scores ])

        rouge2 = np.array([ [score['rouge2'].precision,score['rouge2'].recall, score['rouge2'].fmeasure] for score in scores ])

        return rouge1.mean(axis=0), rouge2.mean(axis=0)

    def compute_sentence_scores(self):
        
        for i in range(len(self)):
            scores = [scorer.score(abstr, self.sentences[i]) for abstr in self.abstracts]
            rouge1 = np.array([ [score['rouge1'].precision,score['rouge1'].recall, score['rouge1'].fmeasure] for score in scores ])
            self.sents_rouge1[i] = rouge1.mean(axis=0)

            rouge2 = np.array([ [score['rouge2'].precision,score['rouge2'].recall, score['rouge2'].fmeasure] for score in scores ])
            self.sents_rouge2[i] = rouge2.mean(axis=0)


    def greedy_select1(self, budget  = 105, scorer = hm_scorer, r = 0.0):
        assert r >= 0 and r <= 1
        selected = []
        sum_length = 0
        max_rouge = 0.0
        idxs = np.argsort(-self.sents_rouge2[:, 1])
        while sum_length < budget:
            max_score = max_rouge
            cur_id = -1
            for i in idxs:
                if (i in selected or self.num_words[i] < 8 or self.num_words[i] > 55) :
                    continue
                c = selected + [i]
                rouge_1, rouge_2 = self.compute_rouge(c)
                score = scorer(rouge_1[1], rouge_2[1], pow(sum_length + self.num_words[i], r ) )
                _logger.debug("idx={:d}, r:{:.2g}, R1:{:.4g}, R2:{:.4g}, score:{:.4g}|{:.4g} ({:d})".format(
                    i, r, rouge_1[1], rouge_2[1], score, max_score, len(c)
                ))
                if score > max_score and ( sum_length + self.num_words[i] < budget ):
                    max_score = score
                    cur_id = i
            
            if (cur_id == -1):
                return selected, sum_length
            selected.append(cur_id)
            sum_length += self.num_words[cur_id]
            max_rouge = max_score
            
        return selected, sum_length

def process(gtc, group, set_, callback = lambda x: x):
    tikk = time.time()
    processed = {
        'sent': gtc.sentences, 
        'num_words': gtc.num_words, 
        'position': gtc.positions, 
        'group': [group] * len(gtc), 
        'set': [set_] * len(gtc),
        'doc_id': gtc.docids,
        'sent_id': np.arange(len(gtc)),
        'doc_sents': gtc.doc_sents,
        'R1.P': gtc.sents_rouge1[:, 0],
        'R1.R': gtc.sents_rouge1[:, 1],
        'R1.F': gtc.sents_rouge1[:, 2],
        'R2.P': gtc.sents_rouge2[:, 0],
        'R2.R': gtc.sents_rouge2[:, 1],
        'R2.F': gtc.sents_rouge2[:, 2],
        'UnresPPR': gtc.unresolved_pronouns,
        'nouns': gtc.nouns,
        'prpns': gtc.prpns,
    }
    oracles = []
    labels = dict()
    N = len(gtc)
    
    for r in Rs:
        for scorer, f in SCORERS.items():
            tic = time.time()
            y = np.zeros(N, dtype = np.int8)
            _logger.info("processing {}-{}, scorer={}, r = {}".format(group, set_, scorer, r))
            summ_idxs, cost = gtc.greedy_select1(scorer = SCORERS[scorer], r = r)
            r1, r2 = gtc.compute_rouge(summ_idxs)
            row = "%s,%s,%s,%s,%.2g,%d,%d,%.4f,%.4f,%.4f,%.4f,%s"%(
                group, set_, "greedy1", scorer, 
                r, cost, len(summ_idxs),
                round(float(r1[0]),4), round(float(r1[1]),4), 
                round(float(r2[0]),4), round(float(r2[1]),4),
                ":".join([str(i) for i in summ_idxs] )
            )
            oracles.append(row)
            y[summ_idxs] = 1
            labels["y_{}_{}".format(scorer, r)] = y
            _logger.info("processed {}-{}, scorer={}, r={:g} in {:g} secs| R1.R={:.4g} R2.R={:.4g}".format(
                group, set_, scorer, r, time.time() - tic, round(float(r1[1]),4), round(float(r2[1]),4)
            ))
    _logger.info("processed {}-{} in {:.2g} secs".format(group, set_, time.time() - tikk ))
    processed.update(labels)
    callback((oracles, processed))

def main():
    path = sys.argv[1] #"../data/duc04.json"
    n_proc = int(sys.argv[2]) ## n_proc == 1 => just save sents 

    if n_proc >= 1:
        manager = mp.Manager()
        q = manager.Queue()   
        pool = mp.Pool(n_proc)
        watcher = pool.apply_async(listener, (q, path))
    jobs = []

    tmp_path = path.split("/")[-1].replace(".json", "-all.sents")
    ftmp = open(tmp_path, "w")
    cnt = 0
    with open(path, "r") as fp:
        i = 0
        for line in fp:
            data = json.loads(line)
            topic = data["group"] + data.get("set", "")
            models = [ s["text"] for s in data["summaries"] if s["type"] == "models"]
            _logger.info("adding {} to jobs".format(topic))

            gtc = RougeGTCreator(data["articles"], models)
            if n_proc >= 1:
                gtc.compute_sentence_scores()
                jobs.append(pool.apply_async( process, (gtc, data["group"], data.get("set", ""), q.put) ))

            ftmp.write("\n".join(gtc.sentences))
            cnt += len(gtc.sentences)
            ftmp.write("\n")
        
    if n_proc >= 1:
        for job in jobs: 
           job.get()
        q.put('kill')
        pool.close()

    ftmp.close()
    _logger.info("Done, #sents:{}".format(cnt))
            
if __name__ == "__main__":
    main()