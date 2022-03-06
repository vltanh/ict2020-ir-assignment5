import numpy as np
import faiss

DESCRIPTORS_PATH = 'output/descriptors.npy'
OUTPUT_DIR = 'output'
RATIO_CENTERS = 0.01

d = np.load(DESCRIPTORS_PATH)

km = faiss.Kmeans(d.shape[1],
                  int(RATIO_CENTERS * d.shape[0]),
                  verbose=True, gpu=True)
km.train(d)

np.save(f'{OUTPUT_DIR}/kmeans.npy', km.centroids)
