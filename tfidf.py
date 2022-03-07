import json
import numpy as np
import scipy.sparse as sp

CLUSTERID_PATH = 'output/clusterids.json'
OUTPUT_DIR = 'output'

docs = json.load(open(CLUSTERID_PATH))

unique_terms = set()
for fn, doc in docs.items():
    unique_terms.update(doc)

count = sp.lil_matrix((len(docs), len(unique_terms)))

for i, (fn, doc) in enumerate(docs.items()):
    for term in doc:
        count[i, term] += 1

tf = count.tocsr()
tf.data = tf.data / \
    np.repeat(
        np.add.reduceat(tf.data, tf.indptr[:-1]),
        np.diff(tf.indptr)
    )

idf = np.log(len(docs) / tf.getnnz(0))

tfidf = tf @ sp.diags(idf)

sp.save_npz(OUTPUT_DIR + '/tfidf.npz', tfidf)
np.save(OUTPUT_DIR + '/idf.npy', idf)
