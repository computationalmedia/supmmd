from allennlp.predictors import Predictor
import itertools
from sacremoses import MosesDetokenizer
import spacy
import re
from commons.utils import get_logger
logger = get_logger("SRL")
# NOTE: SRL model available at: https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz

class SentenceCompressor(object):
    def __init__(self, srl_model_path = "./srl-model-2018.05.25.tar.gz"):
        self.nlp = spacy.load("en_core_web_sm")
        self.srl = Predictor.from_path(srl_model_path)
        self.twd = MosesDetokenizer()
        self.rm_numpkt_prefix = re.compile('^(.*?)([a-zA-Z])')

    # get candidate sentences after doing the SRL and removing ARGM-TMP and ARGM-MNR
    def get_srl_candidates(self, sentence):

        # run semantic role labeling 
        result = self.srl.predict(sentence=sentence)

        # find all removal candidates and mark them
        rm_can_idx = 1 # current index of contigous run
        to_remove = [0] * len(result['words']) # 0 is dont remove, 1+ is indices of removal candidates
        for v in result['verbs']:
            for i,t in enumerate(v['tags']):
                # remove when and how phrases
                if t.endswith('ARGM-TMP') or t.endswith("ARGM-MNR"):
                    if i > 0 and to_remove[i-1] != 0:
                        to_remove[i] = to_remove[i-1]
                    else:
                        to_remove[i] = rm_can_idx
                        rm_can_idx+=1
        
        # build the list of candidates (all combinations of possible removals)
        all_candidates = [" ".join(result['words'])]
        for k in range(1, rm_can_idx):
            for chosen in itertools.combinations(range(1, rm_can_idx), k):
                out_words = []
                for i, w in enumerate(result['words']):
                    if to_remove[i] not in chosen:
                        out_words.append(w)
                all_candidates.append(" ".join(out_words))
        
        return all_candidates

    # get candidates by extracting obj subtrees
    # only returns new candidates not the original sentence
    def get_dep_candidates(self, sentence):
        said_words = ["said", "says", "tells", "told", "wrote", "writes", "write", "reported", "reports"]
        doc = self.nlp(sentence)

        res = []
        for tok in doc:
            # tokens which are said words
            if tok.text.lower() in said_words:
                for c in tok.children:
                    # get the obj part
                    if c.dep_.endswith("obj") or c.dep_ == "ccomp":
                        stree = " ".join(map(str, list(c.subtree)))
                        res.append(stree)
        return res

    def filter_candidates(self, candidates):
        # get the longest 10 candidates
        ncans = sorted(candidates, key = lambda x: -len(x))[:10]

        # remove candidates with < 10 chars or <= 5 words or <= half orig # words
        orig_sent_words = len((ncans[0].split()))
        ncans = filter(lambda x: len(x) > 10 and len(x.split()) > 5 and len(x.split()) >= orig_sent_words//2, ncans)

        # remove duplicates
        ncans = list(set(ncans))

        return ncans

    def detok_post_process(self, sent):
        # remove bracketed text
        f_rmb = self.strip_bracket(sent)
        
        # replace prefixes that are not alphabetic
        f_rmb = re.sub('^(.*?)([a-zA-Z])', '\\2', f_rmb)

        # uppercase the first letter and add a full stop at the end
        if f_rmb[-1] not in ['.', '!', '?']:
            fstop = '.'
        else:
            fstop = ''
        f_rmb = f_rmb[0].upper() + f_rmb[1:] + fstop

        # do detokanization
        f_res = self.twd.detokenize(f_rmb.split())

        return f_res


    def compress(self, sentence):

        # remove parts of sentence using SRL
        srl_cans = self.get_srl_candidates(sentence)

        # extract parts using dep parse
        all_cans = []
        for can in srl_cans:
            all_cans.append(can)
            dep_cans = self.get_dep_candidates(can)
            all_cans.extend(dep_cans)

        # filter out short candidates and restrict to longest 10
        f_cans = self.filter_candidates(all_cans)

        # detokenise the candidates and filter if too short
        out_cans = []
        for can in f_cans:
            dcan = self.detok_post_process(can)

            # filter out short candidates
            if len(dcan) > 10:
                out_cans.append(dcan)
        return out_cans

    # remove any text inside backets
    def strip_bracket(self, txt):
        parens = {'(':')', '[':']', '{':'}', '<':'>'}
        pstack = []
        res = []
        for t in txt:
            if t in parens:
                pstack.append(t)
            if not pstack:
                res.append(t)
            if pstack and parens[pstack[-1]] == t:
                pstack.pop()
        return ''.join(res)


def main():
    sentence1 = "A ban against bistros providing plastic bags free of charge will be lifted at the beginning of March."
    sentence2 = "December 19, 2000: Airbus officially launches the plane, calling it the A380."
    sentence3 = "Federal Education Minister Dan Tehan says the Commonwealth and states have agreed that all year 12 students will finish high school this year."
    sentence4 = "The Premier says the number of cases in NSW continues to stabilise, but the number of community transmissions is continuing to increase."
    sentence5 = "December 19, 2000: Airbus officially launches the plane to be released on the 28th July, calling it the A380."
    sentence6 = "I keep being told by people that it’s ok for them to go to their holiday house (here in Victoria) over this Easter period."
    sentence7 = 'He keep telling "I didn\'t do it!!"'
    sentence8 = "The officials said the raid was timed so that more than 70 cruise missiles would hit bin Laden's camps at the moment when the Central Intelligence Agency believed he would be meeting there with his chief operatives."
    sentence9 = '""As we said from the time of the Aug 20 strike, the objective was to disrupt the training, organization and infrastructure of the bin Laden terrorist network at the Khost camps,"" said David C. Leavy, a spokesman for the National Security Council at the White House."'
    sentences = [sentence1, sentence2, sentence3, sentence4, sentence5, sentence6, sentence7, sentence8, sentence9]
    logger.info("loading model")
    sc = SentenceCompressor("/opt/installed/var/models/srl-model-2018.05.25.tar.gz")
    logger.info("model loaded")
    for sentence in sentences:
        candidates = sc.compress(sentence)
        for c in candidates:
            print(c, c == sentence)
        print('--------')
        logger.info("processed " + sentence)

if __name__ == '__main__':
    main()