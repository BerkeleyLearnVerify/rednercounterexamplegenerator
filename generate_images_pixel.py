from pixel_adv import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--img-dir', type=str, help="the relative path to the image directory")
parser.add_argument('--img-list', type=str, help="a text file with all image names in the dir")
parser.add_argument('--save-dir', type=str, help="the path to where you want to save the image")
parser.add_argument('--label', type=int)
parser.add_argument('--target', type=int)
parser.add_argument('--attack', type=str, choices=['FGSM', 'PGD', 'CW'])
#for vgg16, shape is (224,224)

ROOT_DIR = "/Users/lakshyajain/Desktop/rednercounterexamplegenerator/"
args = parser.parse_args()

image_file = open(args.img_list, 'r')
image_names = []
for line in image_file.readlines():
    image_names += [line.strip("\n").split("/")[-1]]

label = args.label
target = args.target
attack_type = args.attack
out_dir = args.save_dir

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
v = PixelPerturb(framework=vgg16, framework_shape=(224,224), normalize_params=vgg_params)
for image_name in image_names:
    img = plt.imread(ROOT_DIR + args.img_dir + "/" + image_name)[:, :, :3]
    pred, img = v.CW(img, iters=5, lr=0.01, label=label)
    #out_dir += "/" + hashcode
    try:
        if attack_type == "FGSM":
            pred, img = v.FGSM(img, iters=5, eps=0.01, label=label)
            plt.imsave(ROOT_DIR + out_dir + "/" + image_name, np.clip(img, 0, 1))
        elif attack_type == "PGD":
            pred, img = v.PGD(img, iters=5, lr=0.1, epsilon=0.025, label=label)
            plt.imsave(ROOT_DIR + out_dir + "/" + image_name, np.clip(img, 0, 1))
        elif attack_type == "CW":
            pred, img = v.CW(img, iters=5, lr=0.005, label=label)
            plt.imsave(ROOT_DIR + out_dir + "/" + image_name, np.clip(img, 0, 1))

        total_errors += (pred != label)
        sample_size += 1
        print("Total Errors: ", total_errors)
        print("Sample Size: ", sample_size)
        print("\n\n\n")

    except Exception as e:
        print("ERROR")
        print(e)
        print("Error, skipping " + image_name)
        continue

if attack_type is not None:
    print("Total number of misclassifications: ", total_errors)
    print("Error rate: ", total_errors/sample_size)
#a note: to insert any other obj detection framework, you must simply load the model in, get the mean/stddev of the data per channel in an image
#and get the index to label mapping (the last two steps are only needed(if not trained on imagenet, which is provided above),
#now, you have a fully generic library that can read in any .obj file, classify the image, and induce a misclassification through the attack alg
