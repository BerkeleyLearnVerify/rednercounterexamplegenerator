from redner_adv import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, help="the shapenet/imagenet ID of the class")
parser.add_argument('--hashcode_file', type=str, help="a text file with a list of shapenet hashcodes")
parser.add_argument('--label', type=int)
parser.add_argument('--target', type=int)
parser.add_argument('--pose', type=str, choices=['forward', 'top', 'left', 'right', 'all'], default='all')
parser.add_argument('--attack', type=str, choices=['FGSM', 'PGD', 'CW'])
parser.add_argument('--params', type=str, choices=["vertex", "pose", "all"], default="all")
#for vgg16, shape is (224,224)

args = parser.parse_args()

hashcode_file = open(args.hashcode_file, 'r')
hashcodes = []
for line in hashcode_file.readlines():
    hashcodes += [line.strip("\n")]

obj_id = args.id
label = args.label
target = args.target
attack_type = args.attack

pose = args.pose
if pose == 'all':
    poses = ['forward', 'top', 'left', 'right']
else:
    poses = [pose]

vertex_attack = args.params == "vertex" or args.params == "all"
pose_attack = args.params == "pose" or args.params == "all"

print("Vertex Attack: ", vertex_attack)
print("Pose Attack: ", pose_attack)

background = "/work/yan.andy4/rednercounterexamplegenerator/lighting/blue_white.png"
imagenet_filename = "/work/yan.andy4/rednercounterexamplegenerator/class_labels.json"

if attack_type is None:
    out_dir = "/work/yan.andy4/rednercounterexamplegenerator/out/benign/" + obj_id
else:
    out_dir = "/work/yan.andy4/rednercounterexamplegenerator/out/" + attack_type + "/" + args.params + "/" + obj_id

#background = "/home/lakshya/rednercounterexamplegenerator/lighting/blue_white.png"
#imagenet_filename = "/home/lakshya/rednercounterexamplegenerator/class_labels.json"

#if attack_type is None:
#    out_dir = "/home/lakshya/rednercounterexamplegenerator/out/benign/" + obj_id
#else:
#    out_dir = "/home/lakshya/rednercounterexamplegenerator/out/" + attack_type + "/" + args.params + "/" + obj_id

#NOTE ANDREW MAKE SURE WE CHANGE THIS BEFORE RUNNING ANY ADV EXAMPLES!!!!!
#changed!
vgg_params = {'mean': torch.tensor([0.6109, 0.7387, 0.7765]), 'std': torch.tensor([0.2715, 0.3066, 0.3395])}

total_errors = 0
sample_size = 0
for hashcode in hashcodes:
    print(hashcode)
    for pose in poses:
        obj_filename = "/work/yan.andy4/rednercounterexamplegenerator/ShapeNetCore.v2/" + obj_id + "/" + hashcode + "/models/model_normalized.obj"
        #out_dir += "/" + hashcode
        try:
            v = SemanticPerturbations(vgg16, obj_filename, dims=(224,224), label_names=get_label_names(imagenet_filename), normalize_params=vgg_params, background=background, pose=pose, attack_type=attack_type)
            if attack_type is None:
                v.render_image(out_dir=out_dir, filename=hashcode + '_' + pose + ".png")
                print("\n\n\n")
                continue
            elif attack_type == "FGSM":
                pred, img = v.attack_FGSM(label, out_dir=None, save_title=None, steps=5, vertex_eps=0.002, pose_eps=0.15,
                                          vertex_attack=vertex_attack, pose_attack=pose_attack)
                plt.imsave(out_dir + "/" + hashcode + '_' + pose + ".png", np.clip(img[0].permute(1, 2, 0).data.cpu().numpy(), 0, 1))
            elif attack_type == "PGD":
                pred, img = v.attack_PGD(label, out_dir=None, save_title=None, steps=5, vertex_epsilon=5.0, pose_epsilon=0.5, vertex_lr=0.01, pose_lr=0.20, vertex_attack=vertex_attack, pose_attack=pose_attack)
                plt.imsave(out_dir + "/" + hashcode + '_' + pose + ".png", np.clip(img[0].permute(1, 2, 0).data.cpu().numpy(), 0, 1))
            elif attack_type == "CW":
                pred, img = v.attack_cw(label, out_dir=out_dir, save_title=hashcode + '_' + pose, steps=5, vertex_lr=0.01, pose_lr=0.20,
                                         vertex_attack=vertex_attack, pose_attack=pose_attack, target=target)
                plt.imsave(out_dir + "/" + hashcode + '_' + pose + ".png", np.clip(img[0].permute(1, 2, 0).data.cpu().numpy(), 0, 1))

            total_errors += (pred.item() != label)
            sample_size += 1
            print("Total Errors: ", total_errors)
            print("Sample Size: ", sample_size)
            print("\n\n\n")

        except Exception as e:
            print("ERROR")
            print(e)
            print("Error, skipping " + hashcode + ", pose " + pose)
            continue

if attack_type is not None:
    print("Total number of misclassifications: ", total_errors)
    print("Error rate: ", total_errors/sample_size)
#a note: to insert any other obj detection framework, you must simply load the model in, get the mean/stddev of the data per channel in an image
#and get the index to label mapping (the last two steps are only needed(if not trained on imagenet, which is provided above),
#now, you have a fully generic library that can read in any .obj file, classify the image, and induce a misclassification through the attack alg
