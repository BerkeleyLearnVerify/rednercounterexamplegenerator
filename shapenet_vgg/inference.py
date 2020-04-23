import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from dataset import ShapeNetRednerDataset
from finetune import AccuracyTracker

NUM_CLASSES = 12
feature_extract = False

model_ft = models.vgg16(pretrained=True)
# set_parameter_requires_grad(model_ft, feature_extract)
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
input_size = 224

print(model_ft)

path = '/nfs/diskstation/andrew_lee/cs294/shapenet_redner_imgs/out/FGSM/vertex'

shape_dataset = ShapeNetRednerDataset(path,
                        transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.6109, 0.7387, 0.7765],
                                                 [0.2715, 0.3066, 0.3395]),
                        ]))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)
# Send the model to GPU
model_ft = model_ft.to(device)

model_ft.load_state_dict(torch.load('model_ft.pt'))
model_ft.eval()

shape_dataloader = torch.utils.data.DataLoader(shape_dataset, batch_size=1,
                                               num_workers=1)
acc_tracker = AccuracyTracker(12, 3, shape_dataset.classes)

for inputs, labels in tqdm(shape_dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    # zero the parameter gradients
    outputs = model_ft(inputs)
    _, preds = torch.max(outputs, 1)
    _, topkpreds = torch.topk(outputs, 3, 1, True, True)
    acc_tracker.update(labels.data.cpu().numpy(),
                       topkpreds.data.cpu().numpy())

acc_tracker.show()
# for i in val_idx:
#     print(shape_dataset.data[i])
#     inputs, labels = shape_dataset[i]
#     inputs = inputs.unsqueeze(0)
#     inputs = inputs.to(device)
#     with torch.set_grad_enabled(False):
#         outputs = model_ft(inputs)
#         print(outputs)

#     im = np.transpose(np.squeeze(inputs.cpu().numpy()), (1, 2, 0))

#     plt.imshow(im)
#     plt.show()
