import re
from commons.utils import get_logger

logger_replace = get_logger("duc_replace")
logger_cleaner = get_logger("duc_cleaner")

# [^A-Za-z\s,\d\.\'"\(\):;$?/!]
DOC_REPLACES = [
    (r'\t', ' '),
    (r'^\s+', ''),
    (r';', ','),
    (r'(\d),(\d)', r'\1\2'), ## 1,500 => 1500,
    (r'^\s*(\d+)\.\s', r'\1) '), ## 1. becomes 1)
    (r'^\s*(Q|A)\.\s', r'\1 ) '), # Q. or A. converts to Q) /A) 
    (r'\&(AMP|amp)\;?', 'and'), ## fix &amp as and
    (r'\&[a-zA-Z]{2,3}\;?[\s\n]?', ' '), ### remove &QT, &ht like escapes
    (r'[`\']{2,}', r'"'),
    (r'`', "'"),
    (r'([Ee])-(Mm)(Aa)(Ii)(Ll)', r'\1mail'),
    (r'([^a-zA-Z])([Mm]r|[Mm]rs|[Mm]s|[Dd]r|[Ss]ens{0,1}|[Cc]ol|[Rr]ep)\.', r'\1\2 '),
    (r'([Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ept{0,1}|[Oo]c+t|[Nn]ov|[Dd]ec)\.\s*(\d)', r'\1 \2'), ## Nov. 8 => Nov 8
    (r'^\s*\(([a-z]+\s+)+[a-z]+\)[\s\n]+([A-Z\d"])', r' \2'), ## (djw aos)TRIPOLI
    (r'\n+', '\n'),
    (r'[-_#\]\[\|\{\}]+', ' '), ## remove ----- like things
    # (r'\s\'[A-Za-z\d]', ''),
    # (r'(MOVED|WITH|INTERNET\sLINKS|NATIONAL)\n', ''),
]

INVALID_LINES = [
    r'^[^a-z]+ENDS?\s+HERE[^a-z]+',
    r'^[^a-z]+OPTIONAL[^a-z]+',
    r'^[^a-z]+REACH\s+US[^a-z]+'
    r'^[^a-z]+END\sOF\sSTORY[^a-z]+',
    r'(X\s?){3,}',
    r'^[^a-zA-Z]+$', ## lines without alphabets
    r'^nn',
    r'^\s*MORE\s*nn\s*$',
    r'^\s*(ADDI|OP)TIONAL\s*TRIM\s*$',
    r'Story\scan\send\shere\.\sOptional\sadds\sfollow',
    r'^\s+$', ## clank lines
    # r'^[^\"\'\s]+[^\?\.][\"\']?$', ## ## filter out lines with single token that is not valid word
]

LINE_REPLACES = [
    (r'^[A-Za-z\d\,\.\s]+\s*\([a-zA-Z]+\)\s+[-_]*\s([A-Z])', r'\1'), ## for things like BATTLE CREEK, Mich. (AP) -- 
    (r'\.?[^a-z]+ENDS?\s+HERE[^a-z]+', ''),
    (r'^[\s-]+', ''), ### - - - 
    (r'^nn\s', ''), ## some valid sentencs begin with nn
    (r'\s*(nn|HH)\s*$', r''), ## sentences ending with nn
    (r'\&[a-zA-Z]{2,3}\;?[\s\n]?', ''), ### remove &QT, &ht like escapes
    (r'(brk|tx)\:', ''), ## don't know what these are
    (r'www\.\s+(.)', r'www.\1'), ## some urls have space in between
    (r'\s\_+\s', ' '), ## remove _like things
    (r'(\.\s*){2,}', ','), ## fix . . . 
    (r'(\`\`|\'\'|\“|\”)', '"'), ## normalize quotes
    (r'\s{2,}', ' '), # remove unnecessary spaces
]

LINE_FIXERS = [
    (r'([\;\,])\n\s*([a-zA-Z\d\"\'])', r'\1 \2'), ## fixing lines ending by ; or ,
    (r'^([A-Z\d][^\n]+[^\.\?\>\"\'])\s*\n+\s*([a-z\-]+\s[^\n]+[\.\?\"\'])$', r'\1 \2'), #fixing broken sentences
    (r'(Diseases)\n(Control)', r'\1 \2'),
    (r'(editor)\n(Lewis)', r'\1 \2'),
    (r'(writer)\n(Harrison)', r'\1 \2'),
    (r'(GOP)\n(Andy)', r'\1 \2'),
    (r'(by)\n(Sharon)', r'\1 \2'),
    (r'(Kate)\n(Jackson)', r'\1 \2'),
    (r'(topic)\n(Displaying)', r'\1 \2'),
    (r'(process)\n(But)', r'\1 \2'),
    (r'(Horse)\n(It)', r'\1 \2'),
    (r'(Vouge\'s)\n(Anna)', r'\1 \2'),
    (r'(Home)\n(Attorney)', r'\1 \2'),
]

DOC_REPLACES = [(re.compile(pat, flags = re.MULTILINE), sub) for pat, sub in DOC_REPLACES]
INVALID_LINES = [re.compile(pat) for pat in INVALID_LINES]
LINE_REPLACES = [(re.compile(pat), sub) for pat, sub in LINE_REPLACES]
LINE_FIXERS = [(re.compile(pat, flags = re.MULTILINE), sub) for pat, sub in LINE_FIXERS]

def clean_doc(txt):
    for seq in range(5):
        for rex, sub in DOC_REPLACES:
            search_ = rex.search(txt)
            if search_ is not None:
                logger_cleaner.info("<doc-fix{}: {} > : {}".format(seq, rex, search_.group(0)))
            txt = rex.sub(sub, txt)
    return txt

def clean_line(txt, headline = False):
    txt = txt.strip()
    ## replace things
    for rex, sub in LINE_REPLACES:
        search_ = rex.search(txt)
        if search_ is not None:
            logger_replace.info("<replace-rex: {} > : {}".format(rex, search_.group(0)))
        txt = rex.sub(sub, txt)
    ## remove invalid lines   
    for rex in (INVALID_LINES if not headline else INVALID_LINES[:-1]) :
        if rex.search(txt) is not None:
            logger_cleaner.info("<spam-rex: {} > : {}".format(rex, txt))
            return None
    if len(txt) == 0: return None
    return txt

def merge_broken_lines(txt, seq = 1):
    txt = txt.strip()
    for rex, sub in LINE_FIXERS:
        search_ = rex.search(txt)
        if search_ is not None:
            logger_replace.info("<sent-fix:{}: {} > : {}".format(seq, rex, search_.group(0)))
        txt = rex.sub(sub, txt)
    # pattern = re.compile(r'^([A-Z\d][^\n]+[^\.\?\>\"\'])\n+([^\s]+[^\.\)]\s[^\n]+[\.\?\"\'])$', flags = re.MULTILINE)
    return txt


def process_summary(txt):
    txt = txt.replace('\n', ' ')
    txt = clean_doc(txt)
    txt = re.sub(r'\s{2,}', ' ', txt)
    return txt

def clean_desc_title(description, title):
    # description = unicodedata.normalize('NFKD', description).encode('ascii', 'ignore').decode('utf8')
    description = description.encode('ascii', 'ignore').decode('utf8')
    description = clean_doc(description)
    description = re.sub(r';', ',', description, flags = re.M | re.U)
    description = merge_broken_lines(description, 1)
    ## break into paragraphs and process text
    logger_cleaner.debug("breaking into paragraphs")
    
    paragraphs = []
    for para in description.split('\n'):
        para = clean_line(para)
        if para is not None and len(para) > 0:
            paragraphs.append(para.strip())
    
    ## merge again after fixing/removing lines
    description = "\n".join(paragraphs).strip()
    description = re.sub(r'\n{2,}', r'\n', description, flags = re.MULTILINE)
    description = merge_broken_lines(description.strip(), 2)
    description = re.sub(r'([a-zA-Z\d])\n+([a-zA-Z\d])', r'\1 \2', description, flags = re.MULTILINE)

    # ## extract headline
    description = clean_doc(description)
    try:
        title = clean_line(title, headline = True)
        title = clean_doc(title)
        title = re.sub(r'\n+', r' ', title, re.MULTILINE)
    except:
        title = ""
    return description, title