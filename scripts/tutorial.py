from redner_adv import *
import torch
import torchvision.transforms as transforms
import torchvision.models.vgg as vgg
import os

NUM_CLASSES = 12
vgg16 = vgg.vgg16(pretrained=True)
num_ftrs = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)

if not torch.cuda.is_available():
    vgg16.load_state_dict(torch.load('torch_models/model_ft.pt', map_location=lambda storage, location: storage))
else:
    vgg16.load_state_dict(torch.load('torch_models/model_ft.pt'))

label_names = get_label_names("class_labels.json")
vgg_retrained_params = {'mean': torch.tensor([0.6109, 0.7387, 0.7765]), 'std': torch.tensor([0.2715, 0.3066, 0.3395])}
obj_filename = "motorcycle_demo_model/models/model_normalized.obj"
benign = SemanticPerturbations(vgg16, obj_filename, dims=(224,224), label_names=label_names, 
                          			normalize_params=vgg_retrained_params, background="blue_white.png", pose="left", num_classes=NUM_CLASSES, attack_type="benign")
benign.render_image(out_dir="demo_out/", filename="motorcycle_demo_left_benign.png")

if not os.path.exists('./demo_out'):
	os.mkdir('./demo_out')

print("\n\n\nFGSM attack with 'left' pose\n\n\n")
# FGSM attack on motorcycle, pose "left"
fgsm_left = SemanticPerturbations(vgg16, obj_filename, dims=(224,224), label_names=label_names, 
                          			normalize_params=vgg_retrained_params, background="blue_white.png", pose="left", num_classes=NUM_CLASSES, attack_type="FGSM")

pred, img = fgsm_left.attack_FGSM(label=7, out_dir="demo_out/", save_title="motorcycle_demo_left_fgsm.png", steps=5, vertex_eps=0.002, pose_eps=0.15, 
									pose_attack=True, vertex_attack=True)

print("\n\n\nFGSM attack with 'right' pose\n\n\n")
# FGSM attack on motorcycle, pose "right"
fgsm_right = SemanticPerturbations(vgg16, obj_filename, dims=(224,224), label_names=label_names, 
                          			normalize_params=vgg_retrained_params, background="blue_white.png", pose="right", num_classes=NUM_CLASSES, attack_type="FGSM")

pred, img = fgsm_right.attack_FGSM(label=7, out_dir="demo_out/", save_title="motorcycle_demo_right_fgsm.png", steps=5, vertex_eps=0.002, pose_eps=0.15, 
									pose_attack=True, vertex_attack=True)

print("\n\n\nCW attack with 'forward' pose\n\n\n")
# Carlini-Wagner attack on motorcycle, pose "top"
cw_right = SemanticPerturbations(vgg16, obj_filename, dims=(224,224), label_names=label_names, 
                          			normalize_params=vgg_retrained_params, background="blue_white.png", pose="forward", num_classes=NUM_CLASSES, attack_type="CW")

pred, img = cw_right.attack_cw(label=7, out_dir="demo_out/", save_title="motorcycle_demo_top_cw.png", steps=5, vertex_lr=0.01, pose_lr=0.45, 
									pose_attack=True, vertex_attack=True)

print("\n\n\nPGD attack with 'top' pose\n\n\n")
# PGD attack on motorcycle, pose "forward"
cw_right = SemanticPerturbations(vgg16, obj_filename, dims=(224,224), label_names=label_names, 
                          			normalize_params=vgg_retrained_params, background="blue_white.png", pose="top", num_classes=NUM_CLASSES, attack_type="PGD")

pred, img = cw_right.attack_PGD(label=7, out_dir="demo_out/", save_title="motorcycle_demo_forward_pgd.png", steps=5, vertex_epsilon=0.5, pose_epsilon=0.25,
                                            vertex_lr=0.01, pose_lr=0.35,pose_attack=True, vertex_attack=True)