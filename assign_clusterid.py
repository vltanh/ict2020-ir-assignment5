import csv
import json

import numpy as np
import faiss
from tqdm import tqdm

METADATA_PATH = 'output/metadata.csv'
DESCRIPTORS_PATH = 'output/descriptors.npy'
KMEANS_PATH = 'output/kmeans.npy'
OUTPUT_DIR = 'output'

# Load descriptors pool
descriptors = np.load(DESCRIPTORS_PATH)

# Create a search index
index = faiss.IndexFlatL2(descriptors.shape[1])
index.add(np.load(KMEANS_PATH))

# Store "terms" (cluster id)
metadata = list(csv.reader(open(METADATA_PATH)))
i = 0
data = dict()
for fn, cnt in tqdm(metadata):
    cnt = int(cnt)

    _, I = index.search(descriptors[i:i+cnt], 1)

    data[fn] = I.reshape(-1).tolist()

    i += cnt
json.dump(data, open(f'{OUTPUT_DIR}/clusterids.json', 'w'))
