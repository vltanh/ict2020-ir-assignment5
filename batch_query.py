import os
import csv

import scipy.sparse as sp
import numpy as np
import cv2 as cv
import faiss
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import extract_descriptors

IMG_DIR = 'data/oxbuild_images'
QUERY_PATH = 'data/query.txt'
NUM_RESULT = 5
OUTPUT_DIR = 'result'

KMEANS_PATH = 'output/kmeans.npy'
TFIDF_PATH = 'output/tfidf.npz'
IDF_PATH = 'output/idf.npy'
METADATA_PATH = 'output/metadata.csv'


def assign_clusterid(cluster_index, descriptors):
    # Search for the nearest centers of each descriptor
    _, I = cluster_index.search(descriptors, 1)
    return I.reshape(-1)


def calculate_single_tf(n_terms, terms):
    # Create an initial zero vector
    tf = np.zeros(n_terms, dtype=np.float32)

    # Count the frequency of each terms
    unique_terms, cnts = np.unique(terms, return_counts=True)
    tf[unique_terms] += cnts

    # Normalize
    tf = tf / tf.sum()

    return tf


def calculate_single_tfidf(tf, idf):
    # TFIDF[doc, term] = TF[doc, term] * IDF[term]
    return tf * idf


def prepare_cluster_index(kmeans_path):
    # Load the k-means clusters
    kmeans_clusters = np.load(kmeans_path)

    # Build FAISS index using L2 distance
    cluster_index = faiss.IndexFlatL2(kmeans_clusters.shape[1])

    # Add the clusters to the index
    cluster_index.add(kmeans_clusters)

    return cluster_index


def load_idf(idf_path):
    return np.load(idf_path).astype(np.float32)


def load_metadata(metadata_path):
    return list(csv.reader(open(metadata_path)))


def prepare_gallery_index(tfidf_path):
    # Load the TF-IDF database
    tfidf = sp.load_npz(tfidf_path)
    tfidf = tfidf.toarray().astype(np.float32)

    # Normalizing each row
    tfidf /= np.sqrt((tfidf ** 2).sum(1, keepdims=True))

    # Build FAISS index using cosine distance
    gallery_index = faiss.IndexFlatIP(tfidf.shape[1])

    # Add the gallery to the index
    gallery_index.add(tfidf)

    return gallery_index


def query(query_path, n_result, cluster_index, idf, gallery_index):
    # Load the image
    img = cv.imread(query_path)

    # Extract the descriptors
    descriptors = extract_descriptors(img)

    # Assign cluster id to the descriptors based on nearest centers
    terms = assign_clusterid(cluster_index, descriptors)

    # Calculate the query TF
    tf = calculate_single_tf(idf.shape[0], terms)

    # Calculate the query TF-IDF
    tfidf = calculate_single_tfidf(tf, idf)

    # Normalize the TF-IDF vector (for cosine)
    tfidf = tfidf / np.sqrt((tfidf ** 2).sum())

    # Search for the query
    D, I = gallery_index.search(tfidf[None], n_result)
    I = I.reshape(-1)
    D = D.reshape(-1)

    return D, I


def show_image(ax, path, title=None):
    img = cv.imread(path)
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


if __name__ == '__main__':
    # Preparation
    cluster_index = prepare_cluster_index(KMEANS_PATH)
    gallery_index = prepare_gallery_index(TFIDF_PATH)
    idf = load_idf(IDF_PATH)
    metadata = load_metadata(METADATA_PATH)

    # Load query list
    query_list = list(csv.reader(open(QUERY_PATH)))

    # Make output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fn, *_ in tqdm(query_list):
        query_path = IMG_DIR + '/' + fn

        # Query
        D, I = query(query_path, NUM_RESULT, cluster_index, idf, gallery_index)

        fig, axes = plt.subplots(1, NUM_RESULT + 1,
                                 figsize=(5*(NUM_RESULT + 1), 5), dpi=100)

        show_image(axes[0], query_path)

        # Result
        for t, (i, d) in enumerate(zip(I, D)):
            result_path = metadata[i][0]
            show_image(axes[1+t], result_path, f'd = {d:.04f}')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR + '/' + fn)
        plt.close()
