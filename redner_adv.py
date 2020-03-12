import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models.vgg as vgg
from torch.autograd import Variable
import pyredner
import matplotlib.pyplot as plt
import urllib
import zipfile
import requests
import json
import numpy as np

vgg16 = vgg.vgg16(pretrained=True)

def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

#the below method was sampled, with approval, from Rehan Durrani's work at https://github.com/harbor-ml/modelzoo/
def get_label_names(filename):
    with open(filename, 'r') as f:
        detect_labels = {
            int(key): value for (key, value) in json.load(f).items()
        }
    return detect_labels

class SemanticPerturbations:
    def __init__(self, framework, filename, dims, label_names, normalize_params, envmap_filename):
        self.framework = framework
        self.image_dims = dims
        self.label_names = label_names
        self.framework_params = normalize_params
        
        # self.objects = pyredner.load_obj(filename, return_objects=True)
        self.material_map, mesh_list, self.light_map = pyredner.load_obj(filename)
        for _, mesh in mesh_list:
            mesh.normals = pyredner.compute_vertex_normal(mesh.vertices, mesh.indices)
        vertices = []
        for _, mesh in mesh_list:
            vertices.append(mesh.vertices)
            mesh.vertices = Variable(mesh.vertices, requires_grad=True)
            mesh.vertices.retain_grad()

        material_id_map = {}
        self.materials = []
        count = 0
        for key, value in self.material_map.items():
            material_id_map[key] = count
            count += 1
            self.materials.append(value)
        
        self.shapes = []
        for mtl_name, mesh in mesh_list:
            #assert(mesh.normal_indices is None)
            self.shapes.append(pyredner.Shape(\
                vertices = mesh.vertices,
                indices = mesh.indices,
                material_id = material_id_map[mtl_name],
                uvs = mesh.uvs,
                normals = mesh.normals,
                uv_indices = mesh.uv_indices))
        
        self.camera = pyredner.automatic_camera_placement(self.shapes, resolution=(512,512))
        # Compute the center of the teapot
        self.center = torch.mean(torch.cat(vertices), 0)
        self.translation = torch.tensor([0., 0.0, 0.], device = pyredner.get_device(), requires_grad=True)
        self.euler_angles = torch.tensor([0., 0., 0.], device = pyredner.get_device(), requires_grad=True)
        self.light = pyredner.PointLight(position = (self.camera.position + torch.tensor((0.0, 0.0, 100.0))).to(pyredner.get_device()),
                                                intensity = torch.tensor((20000.0, 30000.0, 20000.0), device = pyredner.get_device()))
        background = pyredner.imread(envmap_filename)
        self.background = background.to(pyredner.get_device())
        
    # image: the torch variable holding the image
    # net_out: the output of the framework on the image
    # label: an image label (given as an integer index)
    # returns: the gradient of the image w.r.t the given label
    def _get_gradients(self, image, net_out, label):
        score = net_out[0][label]
        score.backward(retain_graph=True)
        #return image.grad

    # classifies the input image 
    # image: np array of input image
    # label: correct class label for image
    def classify(self, image, label):
        self.framework.eval()
        #transform image before classifying by standardizing values
        mean, std = self.framework_params["mean"], self.framework_params["std"]
        normalize = transforms.Normalize(mean, std)
        image = normalize(image.cpu()[0])
        image = image.unsqueeze(0)

        #forward pass
        fwd = self.framework.forward(image)
        
        #classification via softmax
        probs = torch.nn.functional.softmax(fwd[0], dim=0).data.numpy()
        top3 = np.argsort(probs)[-3:][::-1]
        labels = [(self.label_names[i], probs[i]) for i in top3]
        print(labels)
        prediction_idx = top3[0]
        
        #prediction_idx = int(torch.argmax(fwd[0]))
        #prediction = self.label_names[prediction_idx] 
        return prediction_idx, fwd

    # You might need to combine the detector and the renderer into one class...this will enable you to retrieve gradients of the placement w.r.t the input stuff

    # model the scene based on current instance params
    def _model(self):
        # Get the rotation matrix from Euler angles
        rotation_matrix = Variable(pyredner.gen_rotate_matrix(self.euler_angles), requires_grad=True)
        rotation_matrix.retain_grad()
        print("Rotation")
        print(rotation_matrix)
        # Shift the vertices to the center, apply rotation matrix,
        # shift back to the original space, then apply the translation.

        # for _, mesh in self.mesh_list:
        #     mesh.vertices = (mesh.vertices - self.center) @ torch.t(rotation_matrix) + self.center + self.translation
        #     mesh.vertices.retain_grad()
        #     mesh.normals = pyredner.compute_vertex_normal(mesh.vertices, mesh.indices)

        for shape in self.shapes:
            shape.vertices = (shape.vertices - self.center) @ torch.t(rotation_matrix) + self.center + self.translation
            shape.vertices.retain_grad()
            shape.normals = pyredner.compute_vertex_normal(shape.vertices, shape.indices)

        # Assemble the 3D scene.
        scene = pyredner.Scene(camera=self.camera, shapes=self.shapes, materials=self.materials)
        # Render the scene.
        img = pyredner.render_deferred(scene, lights=[self.light], alpha=True)
        return img

    # render the image properly and downsample it to the right dimensions
    def render_image(self):
        img = self._model()
        alpha = img[:, :, 3:4]
        img = img[:, :, :3] * alpha + self.background * (1 - alpha)
        # Visualize the initial guess
        eps = 1e-6
        img = torch.pow(img + eps, 1.0/2.2) # add .data to stop PyTorch from complaining
        img = torch.nn.functional.interpolate(img.T.unsqueeze(0), size=self.image_dims, mode='bilinear')
        print(torch.max(img))
        print(torch.min(img))
        img.retain_grad()
        return img

    # does a gradient attack on the image to induce misclassification. if you want to move away from a specific class
    # then subtract. else, if you want to move towards a specific class, then add the gradient instead.
    def attack_FGSM(self):
        # classify 
        eps = 1e-5
        img = self.render_image()
        plt.imsave("out_images/base.png", img[0].T.data.cpu().numpy())
        for i in range(25):
            pred, net_out = self.classify(img, 899)
            # get gradients
            self._get_gradients(img.cpu(), net_out, 899)
            eps = 1e-6
            print("Hello")
            #print(len(self.mesh_list))
            count = 0
            for shape in self.shapes:
                if not torch.isfinite(shape.vertices.grad).any() or torch.isnan(shape.vertices.grad).any():
                    count += 1
                else:
                    shape.vertices -= torch.sign(shape.vertices.grad/(torch.norm(shape.vertices.grad) + eps)) * eps
            print(count)
            #self.translation = self.translation - self.translation.grad/torch.norm(self.translation.grad) * learning_rate
            #self.translation.retain_grad()
            img = self.render_image()
            plt.imsave("out_images/img_test_" + str(i) + ".png", img[0].T.data.cpu().numpy())
        final_pred, net_out = self.classify(img, 899)
        print(final_pred)
        #print(class_names[final_pred])
        #plt.imsave("img_test.png", img[0].T.data.cpu().numpy())


#for vgg16, shape is (224,224)
envmap_filename = "lighting/blue_white.png"
imagenet_filename = "imagenet_labels.json"
vgg_params = {'mean': torch.tensor([0.485, 0.456, 0.406]), 'std': torch.tensor([0.229, 0.224, 0.225])}
obj_filename = "teapot/teapot.obj"
#obj_filename = "/home/lakshya/ShapeNetCore.v2/02958343/8fadf13734ff86b5f9e6f9c7735c6b41/models/model_normalized.obj"
obj_filename = "/home/lakshya/ShapeNetCore.v2/02958343/8fc3cde1054cc1aaceb4167db4d0e4de/models/model_normalized.obj"
v = SemanticPerturbations(vgg16, obj_filename, dims=(224,224), label_names=get_label_names(imagenet_filename), normalize_params=vgg_params, envmap_filename=envmap_filename)
v.attack_FGSM()


#a note: to insert any other obj detection framework, you must simply load the model in, get the mean/stddev of the data per channel in an image 
#and get the index to label mapping (the last two steps are only needed(if not trained on imagenet, which is provided above),
#now, you have a fully generic library that can read in any .obj file, classify the image, and induce a misclassification through the attack alg
