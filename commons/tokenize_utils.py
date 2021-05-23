from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from commons.utils import get_logger
import re, os, pickle
import commons
from nltk.corpus import wordnet
import traceback

_logger = get_logger("Tokenizer")

STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.lower().startswith('j'):
        return wordnet.ADJ
    elif nltk_tag.lower().startswith('v'):
        return wordnet.VERB
    elif nltk_tag.lower().startswith('n'):
        return wordnet.NOUN
    elif nltk_tag.lower().startswith('r'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = list(map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged))
    lemmatized_sentence = []
    # print(nltk_tagged, wordnet_tagged)
    for word, tag in wordnet_tagged:
        # print(word, tag)
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(LEMMATIZER.lemmatize(word, tag))
    return lemmatized_sentence


def word_tokenize1(sent, stemming = True, stop_words = True, lemmatize = False):
    assert not (stemming and lemmatize), "both stemming and lemmatizer can't be true"
    txt = re.sub(r'\S{2,}@\S{2,}\.\S{2,}(\.\S{2,})?', ' _EMAIL_ ', sent)
    txt = re.sub(r'(http:[^\s]+)', ' _URL_ ', txt)
    txt = re.sub(r'(www\.[^\s]+)', ' _URL_ ', txt)
    txt = re.sub(r'[^\s,]+\.(com?|net|org|it)[^\s,]*', ' _URL_ ', txt)
    txt = re.sub(r'(\$|euro)(\d+\.\d+|\d+)', '_MONEY_', txt)
    txt = re.sub(r"'(\d+)", r'\1', txt)
    txt = re.sub(r'(\d+\.\d+|\d+)', '_NUM_', txt)
    txt = re.sub(r'_NUM_[sS]([^t])', r'_NUMS_\1', txt) ## preserve 1980s, 70s, etc
    txt = re.sub(r'[^a-zA-Z\d\.\'\s/,&_]', ' ', txt) # remove other chars
    txt = re.sub(r'\s{2,}', ' ', txt)
    txt = re.sub(r"(([A-Z]\.){1,}[A-Z]+)\.?", lambda m: m.group(0).replace('.', ''), txt)
    txt = re.sub(r"([^a-zA-Z'\d])'([a-zA-Z\d])", r'\1 \2', txt)
    txt = re.sub(r"[^a-zA-Z\d,\s'_&]", ' ', txt) ## remove . and /
    txt = re.sub(r'\s{2,}', ' ', txt)
    txt = txt.lower().strip()

    if lemmatize:
        words = lemmatize_sentence(txt)
    else:
        words = nltk.word_tokenize(txt)
    if stop_words:
        words = [w for w in words if w not in STOPWORDS]
    if stemming:
        words = [STEMMER.stem(w) for w in words]
    return words