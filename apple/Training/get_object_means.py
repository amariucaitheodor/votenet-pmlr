import numpy as np
from os import listdir
from os.path import isfile, join

dir = 'label/'
onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

sizes = {}
counts = {}

for file in onlyfiles:
    label = np.load(join(dir, file), allow_pickle=True).item()
    for obj, box in zip(label['types'], label['bboxes']):
        if obj not in sizes:
            sizes[obj] = [0, 0, 0]
            counts[obj] = 0
        sizes[obj] += box[3:6]
        counts[obj] += 1

for label, size in sizes.items():
    print(label, size/counts[label])
