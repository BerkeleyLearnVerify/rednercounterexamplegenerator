import json
import os, sys

import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


path = sys.argv[1]

preds_json = json.load(open(path, 'r'))

preds = [p['preds'][0] for p in preds_json]
labels = [p['label'] for p in preds_json]

M = confusion_matrix(labels, preds).astype('float32')

for i in range(M.shape[0]):
    M[i,:] =  M[i,:] / np.sum(M[i,:])

print(M)
print(np.sum(M, axis=1), np.sum(M, axis=0))
print(np.sum(M[7,:]))

classes = ['airplane', 'bench', 'trashcan', 'bus', 'car', 'helmet', 'mailbox',
           'motorcycle', 'skateboard', 'tower', 'train', 'boat']

plt.figure(figsize=(8,8))
plt.matshow(M, cmap='coolwarm', fignum=1)
plt.xticks(range(12), classes, rotation=45, size='small')
plt.yticks(range(12), classes, size='small')

for (i, j), z in np.ndenumerate(M):
    plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')

plt.savefig('/'.join(path.split('/')[:-1] + ['confusion_matrix.png']))
