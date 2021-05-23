import nltk
import json, pickle
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters, PunktTrainer

ABBRS = ["N.Y", "Corp", "Co", "No", "al", "Blvd", "Ltd", "Roe", "U.S", "Ave", "B.C",
			"Pa", "Mass", "Q", "A", "B", "ACLU", "L", "E", "Act", "A.Q", "R", "D",
            "A.N.C", "D.C", "C", "Inc",
 		]

paths = ["../data/duc03.json", "../data/duc04.json", "../data/tac08_icsi.json", "../data/tac09_icsi.json"]
tok = PunktSentenceTokenizer()
trainer = PunktTrainer()
texts = []
for path in paths:
    with open(path, "r") as fp:
        i = 0
        for line in fp:
            data = json.loads(line)
            for article in data["articles"]:
                text = article["body"]["text"].rstrip().replace('\n', ' ').replace(r'\s{2,}', ' ')
                trainer.train(text,  finalize=False)
            print("added {}:{} to training".format(
            	path.split("/")[-1].split(".")[-2], data["group"] + data.get("set", ""))
           	)

trainer.finalize_training(verbose=True)
punkt_params = trainer.get_params()
tok = PunktSentenceTokenizer(punkt_params)
for abbr in ABBRS:
	tok._params.abbrev_types.add(abbr.lower())
print(punkt_params.abbrev_types)
print(punkt_params.collocations)
print(punkt_params.sent_starters)

f = open('../commons/duc_tac_punkt.pik', 'wb')
pickle.dump(tok, f)
f.close()