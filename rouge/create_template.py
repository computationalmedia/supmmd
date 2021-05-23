import json, os, glob, re
import sys

data = sys.argv[1]
summaries_dir = sys.argv[2]
assert data in {"duc03", "duc04", "tac08", "tac09"}

MODELS_PATH = "../rouge/models/%s/"%data
pattern = "*.M.100.*.*"

pattern_model = {
    "duc03": "D3????.M.100.?.*",
    "duc04": "D3????.M.100.?.*",
    "tac08": "D0???-?.M.100.?.*",
    "tac09": "D0???-?.M.100.?.*"
}[data]

MODELS_PATH = os.path.abspath(MODELS_PATH) + "/"
PEERS_PATH = os.path.abspath(summaries_dir) + "/"

models = dict()
topics = set()

for f in glob.glob(MODELS_PATH + "/" + pattern_model):
    filename = f.split("/")[-1]
    s = re.search(r'([dD]\d{4,5}[-AB]*)\.M\.100\.([A-Z])\.([A-Z])', filename)
    topic, id_ = s.group(1), s.group(3)
    topics.add(topic)
    # print(filename, topic, id_)
    if models.get(topic) is None:
        models[topic] = ''
    models[topic] += '<M ID="{}">{}.M.100.{}.{}</M>\n'.format(id_, topic, s.group(2), id_)

topics = sorted(topics)
print("topics: ", topics)
peers = dict()
for f in glob.glob(summaries_dir + "/" + pattern):
    filename = f.split("/")[-1]
    # print(filename)
    s = re.search(r'([dD]\d{4,5}(-[AB])*)\.M\.100\.([A-Z])\.([\.a-zA-Z\d_-]+)', filename)
    topic, some_id, peerid = s.group(1).upper(), s.group(3), s.group(4)
    # print(filename, topic, peerid)
    if topic.upper() in topics:
        if peers.get(topic.upper()) is None:
            peers[topic.upper()] = ''
        peers[topic.upper()] += '<P ID="{}">{}.M.100.{}.{}</P>\n'.format(peerid, topic, some_id, peerid)

template = '<ROUGE_EVAL version="1.5.5">\n'
for topic, model in models.items():
    try:
        t = ""
        t += '<EVAL ID="%s">\n'%topic
        t += '<PEER-ROOT>\n'
        t += PEERS_PATH
        t += '\n</PEER-ROOT>\n'
        t += '<MODEL-ROOT>\n'
        t += MODELS_PATH
        t += '\n</MODEL-ROOT>\n'
        t += '<INPUT-FORMAT TYPE="SPL">\n'
        t += '</INPUT-FORMAT>\n'
        t += '<PEERS>\n'
        t += peers[topic.upper()]
        t += '</PEERS>\n'
        t += '<MODELS>\n'
        t += model
        t += '</MODELS>\n'
        t += '</EVAL>\n'
        template += t
    except:
        pass
template += '</ROUGE_EVAL>'
if summaries_dir[-1] == "/":
    pp = "%s-eval.xml"%( summaries_dir[:-1] )
else:
    pp = "%s-eval.xml"%( summaries_dir )
print("writing to " + pp)
fo = open(pp , "w")
fo.write(template)
fo.close()
