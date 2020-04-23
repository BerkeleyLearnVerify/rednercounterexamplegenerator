import os
import torch
import skimage.io as skio

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class ShapeNetRednerDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.ids = ('02691156', '02828884', '02747177', '02924116', '02958343',
               '03513137', '03710193', '03790512', '04225987', '04460130',
               '04468005', '04530566',)
        self.classes = ('airplane', 'bench', 'trashcan', 'bus', 'car', 'helmet',
                   'mailbox', 'motorcycle', 'skatboard', 'tower', 'train',
                   'boat',)
        self.transform = transform

        self.data = [] # empty list of tuples
        self.labels = [] # empty list of labels

        for i, class_id in enumerate(self.ids):
            class_dir = os.path.join(root_dir, class_id)
            print('Loading class', self.classes[i], 'with', len(os.listdir(class_dir)), 'images')
            gt_label = i

            for p in os.listdir(class_dir):
                self.data.append({
                    'label': i,
                    'path': os.path.join(root_dir, class_id, p)
                })
                self.labels.append(i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx]
        print(data['path'])
        im = skio.imread(data['path'])
        if im.shape[-1] == 4:
            im = im[:,:,:3] # strip alpha channel
        label = data['label']

        if self.transform:
            im = self.transform(im)

        return (im, label)
