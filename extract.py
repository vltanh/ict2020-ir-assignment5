import os
import csv

import numpy as np
import cv2 as cv
from tqdm import tqdm

from utils import extract_descriptors


IMG_DIR = 'data/oxbuild_images'
GALLERY_LIST = 'data/gallery.txt'
OUT_DIR = 'output'
CHECKPOINT_DIR = OUT_DIR + '/checkpoints'
LOAD_CHECKPOINT = True

# Create output directory
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # for checkpointing

# Load checkpoint
if LOAD_CHECKPOINT and len(os.listdir(CHECKPOINT_DIR)):
    latest = sorted(os.listdir(CHECKPOINT_DIR),
                    key=lambda x: int(x.split('_')[0]))[-1]
    latest = int(latest.split('_')[0])

    metadata = list(csv.reader(
        open(f'{CHECKPOINT_DIR}/{latest:0>4}_metadata.csv')))
    all_descriptors = [
        np.load(f'{CHECKPOINT_DIR}/{latest:0>4}_descriptors.npy')]

    done = [x[0].split('/')[-1] for x in metadata]
else:
    metadata = []
    all_descriptors = []
    done = []

gallery_list = sum(csv.reader(open(GALLERY_LIST)), [])
for i, fn in enumerate(tqdm(gallery_list)):
    if fn in done:
        continue

    fp = IMG_DIR + '/' + fn
    try:
        img = cv.imread(fp)
    except:
        print(f'[ERROR] File {fp} is ineligible.')
        continue

    try:
        descriptors = extract_descriptors(img)
    except:
        print(f'[ERROR] Unable to extract descriptors from {fp}.')
        continue

    if descriptors is None:
        print(f'[ERROR] Unable to extract descriptors from {fp}.')
        continue

    metadata.append([fp, len(descriptors)])
    all_descriptors.append(descriptors)

    if (i + 1) % 200 == 0:
        checkpoint = np.vstack(all_descriptors)
        with open(f'{CHECKPOINT_DIR}/{i+1:0>4}_metadata.csv', 'w') as f:
            csv.writer(f).writerows(metadata)
        np.save(f'{CHECKPOINT_DIR}/{i+1:0>4}_descriptors.npy', checkpoint)

all_descriptors = np.vstack(all_descriptors)
with open(f'{OUT_DIR}/metadata.csv', 'w') as f:
    csv.writer(f).writerows(metadata)
np.save(f'{OUT_DIR}/descriptors.npy', all_descriptors)
