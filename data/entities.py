from itertools import chain, product
import numpy as np
import json, sys
from commons.spotlight import entities_list, annotate
from commons.utils import get_logger
logger = get_logger("DBPEDIA-ANNOT")

dataset = sys.argv[1]

spotlight_url = "http://localhost:2222/rest/annotate" #conf.get("dbpedia.spotlight_url")
spotlight_web_url = "http://api.dbpedia-spotlight.org/en/annotate/" ## fallback
MIN_ENTITIES = 2 ## if lower than this in article, try web url

path = "entities_{}.json".format(dataset)
fo = open(path, "w")

with open("{}.json".format(dataset)) as fp:
    for line in fp:
        data = json.loads(line)
        topic = data["group"] + data.get("set", "")
        # print(articles)
        for a in data["articles"]:
            docid = a["docid"]
            try:
                entities = set(entities_list(annotate(a["body"]["text"], url = spotlight_url) ))
                assert len(entities) >= MIN_ENTITIES, "few ents:{}".format(len(entities))
            except Exception as ex:
                logger.warning("{}: {}".format(docid, str(ex)))
                entities = set(entities_list(annotate(a["body"]["text"], url = spotlight_web_url) ))
            logger.info("annotated {}:{}, #ents:{}".format(topic, docid, len(entities)))
            res = {
                    "group": data["group"],
                    "set": data.get("set", ""),
                    "docid": docid,
                    "annotations": list(entities)
            }
            fo.write(json.dumps(res) + "\n" )
            fo.flush()
fo.close()