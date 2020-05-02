import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import os, sys
import copy
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from dataset import ShapeNetRednerDataset


class AccuracyTracker(object):

    def __init__(self, num_classes, k, class_labels=None):
        self.totals = np.zeros((num_classes,))
        self.counts = np.zeros((num_classes, k))
        if class_labels == None:
            class_labels = [i for i in range(num_classes)]
        assert num_classes == len(class_labels)
        self.class_labels = list(class_labels)
        self.predictions = []

    def update(self, labels, topkpreds, example_paths=None):
        """labels is a length `batch_size` vector of labels (indices).
        topkpreds is a size (`batch_size`, `k`) matrix of top-k predictions.
        example_paths is a length k list of paths
        """
        np.add.at(self.totals, labels, 1) # increment label counts
        matches = np.equal(topkpreds, labels[:,np.newaxis])
        for i, l in enumerate(labels):
            # top-1 match counts for top-2 and top-3
            self.counts[l,:] += np.cumsum(matches[i,:])

            self.predictions.append({
                'path': example_paths[i] if example_paths else '',
                'label': int(l),
                'preds': [int(p) for p in topkpreds[i,:]]
            })

    def show(self):
        row_format = "{:>12} " * (len(self.class_labels) + 2) # total column
        headers = ["top-k \ class"] + self.class_labels + ["overall"]
        print(row_format.format(*headers))
        overall = np.sum(self.counts, axis=0) / np.sum(self.totals)
        acc = np.concatenate([self.counts / self.totals[:,np.newaxis],
                              overall[np.newaxis,:]], axis=0)

        acc = np.around(acc, 4)
        for r in range(self.counts.shape[1]):
            print(row_format.format(r + 1, *acc[:, r]))


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    classes = dataloaders['train'].dataset.classes

    acc_trackers = {
        'train': AccuracyTracker(12, 3, classes),
        'val': AccuracyTracker(12, 3, classes)
    }

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    _, topkpreds = torch.topk(outputs, 3, 1, True, True)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                acc_trackers[phase].update(labels.data.cpu().numpy(),
                                           topkpreds.data.cpu().numpy())

            epoch_loss = running_loss / (len(dataloaders[phase]) * dataloaders[phase].batch_size)
            epoch_acc = running_corrects.double() / (len(dataloaders[phase]) * dataloaders[phase].batch_size)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            acc_trackers[phase].show()

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == '__main__':

    dataset_path = sys.argv[1]
    model_path = sys.argv[2]

    NUM_CLASSES = 12
    feature_extract = False

    model_ft = models.vgg16(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
    input_size = 224

    print(model_ft)

    shape_dataset = ShapeNetRednerDataset(dataset_path,
                            transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.6109, 0.7387, 0.7765],
                                                     [0.2715, 0.3066, 0.3395]),
                            ]))

    # Stratified train-val split
    dataset_labels = shape_dataset.labels

    train_idx, test_idx = train_test_split(np.arange(len(dataset_labels)),
                                          test_size=0.3,
                                          shuffle=True,
                                          stratify=dataset_labels,
                                          random_state=100) # DO NOT CHANGE

    val_idx, test_idx = train_test_split(test_idx,
                                         test_size=2/3,
                                         shuffle=True,
                                         stratify=[dataset_labels[i] for i in test_idx],
                                         random_state=100) # DO NOT CHANGE

    print(train_idx)
    print(val_idx)
    print(test_idx)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    shape_train_dataloader = torch.utils.data.DataLoader(shape_dataset, batch_size=4,
                                                         num_workers=4,
                                                         sampler=train_sampler)
    shape_val_dataloader = torch.utils.data.DataLoader(shape_dataset, batch_size=4,
                                                       num_workers=4,
                                                       sampler=val_sampler)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model_ft, hist = train_model(model_ft, {'train': shape_train_dataloader,
                                            'val': shape_val_dataloader },
                                 criterion, optimizer_ft, num_epochs=20)

    torch.save(model_ft.state_dict(), model_path)
