from bs4 import BeautifulSoup
import xmltodict
import re
import json, os, sys, glob
from commons.spotlight import annotate, entities_list
from commons.utils import get_logger, apply_file_handler
import multiprocessing as mp
from collections import defaultdict
from commons.patterns import *

_logger = get_logger("DUC")

def listener(q, path):
    '''listens for messages on the q, writes to file '''
    _logger.info("creating res file: " + path)
    fp = open(path, "w")
    while True:
        m = q.get()
        _logger.DEBUG(json.dumps(m))
        if m == 'kill':
            break
        fp.write(json.dumps(m) + "\n")
        fp.flush()
    fp.close()

# parse a DUC document
def parse_duc_doc(filepath, fp = None):
    with open(filepath, 'r') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    topic_id = filepath.split('/')[-2].strip()
    doc_id = soup.docno.string.strip()

    try:
        body_txt = soup.body.text  ## only available in 2006,2007
    except:
        body_txt = ""    
    text_txt = soup.select('TEXT')[0].text

    ## possibly not inside <text tag>
    if len(text_txt) < 0.5 * len(body_txt):
        _logger.info("{}/{}, extracted from body".format(topic_id, doc_id))
        description = body_txt
    else:
        description = text_txt
    
    try:
        title = soup.headline.string
    except:
        title = ""
    
    description, title = clean_desc_title(description, title)

    if fp is not None:
        fp.write(topic_id + "\t" + doc_id + " << " + title + " >>\n" + "\n".join(
            paragraphs) + "\n" + "=" * 100 + "\n\n")
        fp.flush()
    try:
        date_time = soup.date_time.string.strip()[:10]
    except:
        date_time = "NA" ## only available in 2006,2007

    return doc_id, title, description

def duc_generator(dirname):
    groups = dict()
    _logger.debug("reading files from " + dirname)
    for root, dirs, files in os.walk(dirname):
        for name in files:
            # Exclude hidden files and only cover non-update files
            if not name[0] == ".":
                filename = os.path.join(root, name)
                _logger.debug("reading file: " + name)
                cluster_id = root[-7:-1]
                # set_id = root[-1]
                doc_id, title, text = parse_duc_doc(filename)
                if len(title) < 5 or len(text) < 100:
                    _logger.warning("{}:{}, |title|:{}, |text|:{}".format(
                        cluster_id, filename.split('/')[-1], len(title), len(text)
                ))
                article = {'docid': doc_id, 'title': { "text": title }, 'body': {"text": text}}
                # article = file_to_data(filename)
                if cluster_id in groups:
                    groups[cluster_id] += [article]
                else:
                    groups[cluster_id] = [article]
    
    for k, v in groups.items():
        row = dict()
        row['group'] = k
        row['summaries'] = []
        row['articles'] = v
        yield row

def tac_generator(dirname):
    groups = dict()
    for root, dirs, files in os.walk(dirname):
        for name in files:
            # Exclude hidden files and only cover non-update files
            if not name[0] == ".":
                filename = os.path.join(root, name)
                cluster_id = root[-8:-3]
                set_id = root[-1]
                # print(cluster_id, set_id, filename)
                article = parse_tac_doc(filename)
                if (cluster_id, set_id) in groups:
                    groups[(cluster_id, set_id)] += [article]
                else:
                    groups[(cluster_id, set_id)] = [article]
    
    for k, v in groups.items():
        row = dict()
        row['group'], row['set'] = k[0], k[1]
        row['summaries'] = []
        row['articles'] = v
        yield row

def parse_tac_doc(filename):
    with open(filename, 'r') as fh:
        # transform to real xml
        soup = BeautifulSoup(fh.read(), features='xml')

    cleaned_soup = str(soup).replace('<P>', '\n')
    cleaned_soup = cleaned_soup.replace('</P>', '\n')

    # parse xml and put into ordered dict, remove outside DOC-tag
    article = xmltodict.parse(cleaned_soup)['DOC']
    doc_id = article['@id']

    text = re.sub(r'([^\n])\n([^\n])', r'\1 \2', article.get('TEXT', ''), flags=re.MULTILINE )
    title = article.get('HEADLINE', '')

    text, title = clean_desc_title(text, title)
    data = {'docid': doc_id, 'title': { "text":  title}, 'body': {"text": text}}
    return data

def add_summaries_duc(articles, dirname, pattern):
    """Adds summaries from summary directory to the correct article object."""
    all_summaries = {}
    # D30010.M.100.T.F
    for filename in glob.glob(dirname + pattern, recursive=False):
        splits = filename.split("/")
        cluster_id = splits[-1][:6].lower()

        with open(filename, 'r', encoding="utf8", errors='ignore') as fh:
            try:
                text = process_summary(fh.read())
            except Exception as ex:
                _logger.error("Error reading " + filename)
                raise ex
            summary = {
                "type": splits[-2],
                "text": text,
                "id": splits[-1].split('.')[-1]
            }
        if cluster_id in all_summaries:
            all_summaries[cluster_id] += [ summary ]
        else:
            all_summaries[cluster_id] = [ summary ]
    # print(all_summaries.keys())
    for article in articles:
        article['summaries'] = all_summaries[article['group']]
    return articles

def add_summaries_tac(articles, dirname, pattern):
    """Adds summaries from summary directory to the correct article object."""
    all_summaries = {}
    for filename in glob.glob(dirname + pattern, recursive=False):
        splits = filename.split("/")
        cluster_id = splits[-1][:5]
        set_id = splits[-1][6]
        with open(filename, 'r', encoding="utf8", errors='ignore') as fh:
            try:
                text = process_summary(fh.read())
            except Exception as ex:
                _logger.error("Error reading " + filename)
                raise ex
            summary = {
                "type": splits[-2],
                "text": text,
                "id": splits[-1].split('.')[-1]
            }
        if (cluster_id, set_id) in all_summaries:
            all_summaries[(cluster_id, set_id)] += [ summary ]
        else:
            all_summaries[(cluster_id, set_id)] = [ summary ]
        
    for article in articles:
        article['summaries'] = all_summaries[( article['group'], article['set'] )]
    return articles

def parse_duc():
    fp = open('../data/duc03.json', 'w')
    _logger.info("starting to parse DUC docs/summaries")
    pattern = '/**/D3????.M.100.?.*'
    apply_file_handler(logger_replace, "./log/replace2003.log.txt")
    apply_file_handler(logger_cleaner, "./log/cleaner2003.log.txt")

    data = list(duc_generator("/opt/installed/var/datasets/duc2003/task2/docs/"))
    data = add_summaries_duc(data, "/opt/installed/var/datasets/duc2003/eval_results/", pattern)
    _logger.info("DUC03: #groups=%s"%len(data))
    for row in data:
        row['dataset'] = 'duc03'
        fp.write(json.dumps(row) + "\n")
    fp.close()

    fp = open('../data/duc04.json', 'w')
    apply_file_handler(logger_replace, "./log/replace2004.log.txt")
    apply_file_handler(logger_cleaner, "./log/cleaner2004.log.txt")

    data = list(duc_generator("/opt/installed/var/datasets/duc2004/tasks1and2/"))
    data = add_summaries_duc(data, "/opt/installed/var/datasets/duc2004/duc2004_results/ROUGE/eval/", pattern)
    _logger.info("DUC04: #groups=%s"%len(data))
    for row in data:
        row['dataset'] = 'duc04'
        fp.write(json.dumps(row) + "\n")
    fp.close()

def parse_tac():
    _logger.info("starting to parse TAC docs/summaries")
    fp = open('../data/tac08.json', 'w')
    pattern = '/**/D0???-?.M.100.?.*'
    apply_file_handler(logger_replace, "./log/replace2008.log.txt")
    apply_file_handler(logger_cleaner, "./log/cleaner2008.log.txt")

    data = list(tac_generator("/opt/installed/var/datasets/TAC2008/UpdateSumm08_test_docs_files/"))
    data = add_summaries_tac(data, "/opt/installed/var/datasets/TAC2008/UpdateSumm08_eval/ROUGE/", pattern)
    _logger.info("TAC2008: #groups=%s"%len(data))
    for row in data:
        row['dataset'] = 'tac08'
        fp.write(json.dumps(row) + "\n")
    fp.close()

    fp = open('../data/tac09.json', 'w')
    apply_file_handler(logger_replace, "./log/replace2009.log.txt")
    apply_file_handler(logger_cleaner, "./log/cleaner2009.log.txt")
    data = list(tac_generator("/opt/installed/var/datasets/TAC2009/UpdateSumm09_test_docs_files/"))
    data = add_summaries_tac(data, "/opt/installed/var/datasets/TAC2009/UpdateSumm09_eval/ROUGE/", pattern)
    _logger.info("TAC2009: #groups=%s"%len(data))
    for row in data:
        row['dataset'] = 'tac09'
        fp.write(json.dumps(row) + "\n")
    fp.close()

def _extract(row, threshold, info):
    if len(row["entities"]) < threshold:
        entities = entities_list(annotate( ( row["text"] ) ))
        for concept, surface_form, types in entities:
            if surface_form is None:
                _logger.info("entity={}, surface_form=None".format(concept))
        row["entities"] = entities
        _logger.info("{}: extracted {} entities".format(info, len(entities)))
    return row

def extract_entities_group(line, threshold = 5, callback = lambda x: x):
    row = json.loads(line)
    articles = row["articles"]
    for a in articles:
        a["title"] = _extract(a["title"], threshold = threshold, info = "%s-%s-%s T"%(row["group"], row["set"], a["docid"])) 
        a["body"] = _extract(a["body"], threshold = threshold, info = "%s-%s-%s D"%(row["group"], row["set"], a["docid"]))
    row["articles"] = articles
    summaries = row["summaries"]
    for s in summaries:
        s = _extract(s, threshold = threshold, info = "%s-%s-%s-%s "%(row["group"], row["set"], s["type"], s["id"]))
        summaries.append(s)
    row["summaries"] = summaries

    return callback(row)

def extract_entities():
    rows = []
    infile = sys.argv[3]
    with open(infile, 'r') as fp:
        rows = fp.readlines()
    assert len(rows) == 96
    thresh = int(sys.argv[2])

    manager = mp.Manager()
    q = manager.Queue()    
    # pool = mp.Pool(mp.cpu_count() -1)
    pool = mp.Pool(int(sys.argv[1]))
    watcher = pool.apply_async(listener, (q, infile.replace(".json", "-ents.json")))
    jobs = []
    for line in rows[:10]:
        jobs.append(pool.apply_async(extract_entities_group, (line, thresh, q.put)))
    
    for job in jobs: 
       job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()

    _logger.info("Done")

if __name__ == "__main__":
    # extract_entities()
    arg = sys.argv[1]
    if arg == "duc":
        parse_duc()
    elif arg == "tac":
        parse_tac()
    else:
        raise Exception("arg must be `duc` (for DUC03/04) or `tac` (for TAC08/09)")