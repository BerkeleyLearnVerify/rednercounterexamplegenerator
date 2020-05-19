from redner_adv import *
import torch
import torchvision.transforms as transforms
import torchvision.models.vgg as vgg
import argparse
import os

from verifai.features.features import *
from verifai.samplers.feature_sampler import *
from verifai.falsifier import generic_falsifier
from verifai.monitor import specification_monitor

import numpy as np
from dotmap import DotMap

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

_, mesh_list, _ = pyredner.load_obj(OBJ_FILENAME)

features = {'euler_delta': Feature(Box([-.3, .3]))}
for i, name_mesh in enumerate(mesh_list):
    _, mesh = name_mesh
    features['mesh' + str(i)] = Feature(Array(Box((-.005, .005)), tuple(mesh.vertices.shape)))

space = FeatureSpace(features)
sampler = FeatureSampler.randomSamplerFor(space)

MAX_ITERS = 5
PORT = 8888
MAXREQS = 5
BUFSIZE = 4096

falsifier_params = DotMap()
falsifier_params.n_iters = MAX_ITERS
falsifier_params.compute_error_table = False
falsifier_params.save_error_table = False
falsifier_params.save_safe_table = False
falsifier_params.fal_thres = 0.5

class confidence_spec(specification_monitor):
    def __init__(self):
        def specification(traj):
            return bool(traj['true'] == traj['pred'])
        super().__init__(specification)

server_options = DotMap(port=PORT, bufsize=BUFSIZE, maxreqs=MAXREQS)

falsifier = generic_falsifier(sampler=sampler, server_options=server_options,
                             monitor=confidence_spec(), falsifier_params=falsifier_params)
falsifier.run_falsifier()

