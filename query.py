import csv

import numpy as np
import cv2 as cv
import faiss

from utils import extract_descriptors

QUERY_PATH = 'data/oxbuild_images/all_souls_000026.jpg'
KMEANS_PATH = 'output/kmeans.npy'
TFIDF_PATH = 'output/tfidf.npy'
IDF_PATH = 'output/idf.npy'
METADATA_PATH = 'output/metadata.csv'

img = cv.imread(QUERY_PATH)
descriptors = extract_descriptors(img)

index = faiss.IndexFlatL2(descriptors.shape[1])
index.add(np.load(KMEANS_PATH))
_, I = index.search(descriptors, 1)
I = I.reshape(-1)

idf = np.load(IDF_PATH).astype(np.float32)
emb = np.zeros(idf.shape, dtype=np.float32)
terms, cnts = np.unique(I, return_counts=True)
emb[terms] += cnts
emb = emb / emb.sum()
emb = emb * idf
emb = emb / np.sqrt((emb ** 2).sum())

tfidf = np.load(TFIDF_PATH).astype(np.float32)
tfidf /= np.sqrt((tfidf ** 2).sum(1, keepdims=True))

index2 = faiss.IndexFlatIP(tfidf.shape[1])
index2.add(tfidf)

D, I = index2.search(emb[None], 20)
metadata = list(csv.reader(open(METADATA_PATH)))
metadata = [x[0] for x in metadata]
print(np.array(metadata)[I])
print(D)
