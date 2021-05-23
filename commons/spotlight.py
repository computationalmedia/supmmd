import json, re, traceback, os
import urllib, requests
# from pyhocon import ConfigFactory
from commons.utils import get_logger

# conf = ConfigFactory.parse_file(os.path.dirname(os.path.realpath(__file__)) + "/text.conf")
# spotlight_url = "http://localhost:2222/rest/annotate" #conf.get("dbpedia.spotlight_url")
# spotlight_web_url = conf.get("dbpedia.spotlight_web_url")
support = 2 #conf.get("dbpedia.support")
confidence = 0.4 # conf.get("dbpedia.confidence")
logger = get_logger("spotlight")

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

def entities_list(annotation):
    concepts = []
    if "Resources" in annotation:
        for entity in annotation["Resources"]:
            # types = (tpe for tpe in entity["@types"].split(',') if tpe is not '')
            types = entity["@types"]
            concept = entity["@URI"].split('/')[-1]
            surface_form = entity["@surfaceForm"]
            concepts.append((concept, surface_form.lower(), types))
    return concepts

def annotate(doc, url):
    try:
        doc = emoji_pattern.sub(r'', doc)
        doc = urllib.parse.quote_plus(doc)
        args = {"text": doc.encode("utf-8"), "support": support, "confidence": confidence}
        response = requests.post(
            url,
            data = args,
            headers = {"Accept": "application/json", "content-type":"application/x-www-form-urlencoded"},
            timeout = 30
        )
        return json.loads(response.content.decode('utf-8'))
    except Exception as ex:
        logger.warn("<{}> {}: {}".format(url, response.status_code, response.reason))
        logger.info(str(ex) + "no annotation for:"+ doc)
        return {"@text": doc}
