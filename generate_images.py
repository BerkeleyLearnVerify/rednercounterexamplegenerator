from redner_adv import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str)
parser.add_argument('--hashcode_file', type=str)
parser.add_argument('--label', type=int)
# parser.add_argument('--pose', type=str, choices=['forward', 'top', 'left', 'right'])
parser.add_argument('--attack', type=str, choices=['FGSM'])

#for vgg16, shape is (224,224)

args = parser.parse_args()

hashcode_file = open(args.hashcode_file, 'r')
hashcodes = []
for line in hashcode_file.readlines():
    hashcodes += [line.strip("\n")]

obj_id = args.id
label = args.label
attack_type = args.attack

background = "lighting/blue_white.png"
imagenet_filename = "imagenet_labels.json"

if attack_type is None:
    out_dir = "out/benign/" + obj_id 
else:
    out_dir = "out/" + attack_type + "/" + obj_id

vgg_params = {'mean': torch.tensor([0.485, 0.456, 0.406]), 'std': torch.tensor([0.229, 0.224, 0.225])}

for hashcode in hashcodes:
    for pose in ['forward', 'top', 'left', 'right']:
        obj_filename = "/home/lakshya/ShapeNetCore.v2/" + obj_id + "/" + hashcode + "/models/model_normalized.obj"
        #out_dir += "/" + hashcode
        try:
            v = SemanticPerturbations(vgg16, obj_filename, dims=(224,224), label_names=get_label_names(imagenet_filename), normalize_params=vgg_params, background=background, pose=pose)
            if attack_type is None:
                v.render_image(out_dir=out_dir, filename=hashcode + '_' + pose + ".png")
            elif attack_type == "FGSM":
                v.attack_FGSM(label, out_dir, filename=hashcode + '_' + pose)
        except:
            print("Error, skipping " + hashcode + ", pose " + pose)
            continue

#a note: to insert any other obj detection framework, you must simply load the model in, get the mean/stddev of the data per channel in an image 
#and get the index to label mapping (the last two steps are only needed(if not trained on imagenet, which is provided above),
#now, you have a fully generic library that can read in any .obj file, classify the image, and induce a misclassification through the attack alg
