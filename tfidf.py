import json
import numpy as np

CLUSTERID_PATH = 'output/clusterids.json'
OUTPUT_DIR = 'output'

docs = json.load(open(CLUSTERID_PATH))

unique_terms = set()
for fn, doc in docs.items():
    unique_terms.update(doc)

count = np.zeros((len(docs), len(unique_terms)))

for i, (fn, doc) in enumerate(docs.items()):
    terms, cnts = np.unique(doc, return_counts=True)
    count[i, terms] += cnts

tf = count / count.sum(1, keepdims=True)
idf = np.log(len(docs) / (count > 0).sum(0))

tfidf = tf * idf

np.save(OUTPUT_DIR + '/tfidf.npy', tfidf)
np.save(OUTPUT_DIR + '/idf.npy', idf)
