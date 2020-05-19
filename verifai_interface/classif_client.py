from redner_adv import *
import torch
import torchvision.transforms as transforms
import torchvision.models.vgg as vgg
import argparse
import os

import numpy as np
from dotmap import DotMap

from verifai.client import Client

NUM_CLASSES = 12
OUT_DIR = 'verifai_out'
BACKGROUND = 'lighting/blue_white.png'
IMAGENET_FILENAME = 'class_labels.json'
VGG_PARAMS = {'mean': torch.tensor([0.6109, 0.7387, 0.7765]), 'std': torch.tensor([0.2715, 0.3066, 0.3395])}

# For now we hardcode the object id and hash, and true label 0-11
OBJ_ID = '02958343'
HASHCODE = '7ed6fdece737f0118bb11dbc05ffaa74'
OBJ_FILENAME = '../ShapeNetCore.v2/' + OBJ_ID + '/' + HASHCODE + '/models/model_normalized.obj'
LABEL = 4
POSE = 'forward'


class Classifier(Client):
    def __init__(self, classifier_data):
        port = classifier_data.port
        bufsize = classifier_data.bufsize
        super().__init__(port, bufsize)

        # PREPROCESSING CODE THAT SHAPES THE NETWORK TO OUR SHAPENET DATASET #
        self.vgg16 = vgg.vgg16(pretrained=True)
        num_ftrs = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)

        if not torch.cuda.is_available():
            self.vgg16.load_state_dict(torch.load('torch_models/model_ft.pt', map_location=lambda storage, location: storage))
        else:
            self.vgg16.load_state_dict(torch.load('torch_models/model_ft.pt'))

        self.vgg_params = {'mean': torch.tensor([0.6109, 0.7387, 0.7765]), 'std': torch.tensor([0.2715, 0.3066, 0.3395])}
        self.iters = 0

    def simulate(self, sample):
        with torch.no_grad():
            v = SemanticPerturbations(self.vgg16, OBJ_FILENAME, dims=(224,224), label_names=get_label_names(IMAGENET_FILENAME), 
                                            normalize_params=VGG_PARAMS, background=BACKGROUND, pose=POSE, num_classes=NUM_CLASSES, attack_type='benign')
            print('v init')
            v.euler_angles += torch.tensor(sample.euler_delta, device=pyredner.get_device())
            print('eulers added')
            for i in range(len(v.shapes)):
                v_shape = v.shapes[i].vertices.shape
                v.shapes[i].vertices += torch.tensor(sample._asdict()['mesh' + str(i)], device=pyredner.get_device()).reshape(v_shape)
            print('vertices added')
            img = v.render_image(out_dir=OUT_DIR, filename=HASHCODE + '_' + str(self.iters) + '.png', no_grad=True)
            print('img rendered')
            pred, out = v.classify(img)
            print('forward pass done')
            res = {}
            res['true'] = LABEL
            res['pred'] = pred
            print(res)
            self.iters += 1
            return res

PORT = 8888
BUFSIZE = 4096

classifier_data = DotMap()
classifier_data.port = PORT
classifier_data.bufsize = BUFSIZE
client_task = Classifier(classifier_data)
while True:
    if not client_task.run_client():
        print("End of all classifier calls")
        break
    
