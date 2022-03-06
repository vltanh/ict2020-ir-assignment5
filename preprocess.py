import os
import csv

IMG_DIR = 'data/oxbuild_images'
GT_DIR = 'data/gt_files_170407/'
OUT_DIR = 'data'

query = []
for x in os.listdir(GT_DIR):
    if 'query' not in x:
        continue

    f = open(GT_DIR + '/' + x).readlines()
    fn, *bbox = f[0].split()

    fn = '_'.join(fn.split('_')[1:]) + '.jpg'
    query.append([fn] + bbox)

with open(OUT_DIR + '/' + 'query.txt', 'w') as f:
    csv.writer(f).writerows(query)

query_fn = [x[0] for x in query]
train = list(filter(lambda x: x not in query_fn, os.listdir(IMG_DIR)))
train = [[x] for x in train]
with open(OUT_DIR + '/' + 'gallery.txt', 'w') as f:
    csv.writer(f).writerows(train)
