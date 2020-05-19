import torch
import torchvision
import torchvision.transforms as transforms
import pyredner
from torch.autograd import Variable
import matplotlib.pyplot as plt
import json
import numpy as np
import torch.nn as nn

"""
A function that acts as the hook for the variables in our pipeline. We take all nan gradients and zero them out.
"""
def set_grad(var):
    def hook(grad):
        grad[grad != grad] = 0
        var.grad = grad

    return hook

"""
A helper utility function that takes in a json filename that maps indexes to dataset class names. 
The below method was sampled, with approval, from Rehan Durrani's work at https://github.com/harbor-ml/modelzoo/
"""
def get_label_names(filename):
    with open(filename, 'r') as f:
        detect_labels = {
            int(key): value for (key, value) in json.load(f).items()
        }
    return detect_labels

"""
x: a tensor (torch or numpy)
Simple reduce sum function that takes a tensor and does the reduce sum operation.
"""
def reduce_sum(x, keepdim=True):
    # silly PyTorch, when will you get proper reducing sums/means?
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)

    return x

"""
takes in two tensors, x and y, and computes and returns l2 distance between them
"""
def l2_dist(x, y, keepdim=True):
    d = None
    for x_i, y_i in zip(x, y):
        if d is not None:
            d += torch.sum(reduce_sum((x_i - y_i) ** 2, keepdim=keepdim))
        else:
            d = torch.sum(reduce_sum((x_i - y_i) ** 2, keepdim=keepdim))

    return d

"""
arctanh function for Carlini-Wagner attack
"""
def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5

"""
rescaling the tanh for Carlini-Wagner attack
"""
def tanh_rescale(x, x_min=-1., x_max=1.):
    return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min


class SemanticPerturbations:
    """
    framework: The object classification framework to be used and attacked. 
    filename: the .obj file to read the object from.
    dims: the input image dimension (excluding the # of channels) of the classification framework -- e.g. for VGG16, it's (224,224)
    label_names: the dictionary of indexes/labels -> label names 
    normalize_params: the mean/std. dev of the dataset.
    background: the background image filename to blend the object against.
    pose: for now, we provide 4 choices for object pose: 'left', 'right', 'forward', 'top'
    attack_type: what attack to perform for this framework: 'FGSM', 'CW', 'PGD' are the 3 choices. 
    """
    def __init__(self, framework, filename, dims, label_names, normalize_params, background, pose, num_classes,
                 attack_type="benign"):

        self.NUM_CLASSES = num_classes
        self.framework = framework.to(pyredner.get_device())
        self.image_dims = dims
        self.label_names = label_names
        self.framework_params = normalize_params

        # self.objects = pyredner.load_obj(filename, return_objects=True)
        self.material_map, mesh_list, self.light_map = pyredner.load_obj(filename)
        for _, mesh in mesh_list:
            mesh.normals = pyredner.compute_vertex_normal(mesh.vertices, mesh.indices)

        vertices = []

        self.modifiers = []
        self.input_adv_list = []
        self.input_orig_list = []
        self.targeted = False
        self.clamp_fn = "tanh"

        self.attack_type = attack_type

        if attack_type == "CW":
            for _, mesh in mesh_list:
                vertices.append(mesh.vertices)
                modifier = torch.zeros(mesh.vertices.size(), requires_grad=True, device=pyredner.get_device())
                self.modifiers.append(modifier)
                self.input_orig_list.append(tanh_rescale(torch_arctanh(mesh.vertices)))
                mesh.vertices = tanh_rescale(torch_arctanh(mesh.vertices) + modifier)

                self.input_adv_list.append(mesh.vertices)
                mesh.vertices.retain_grad()
        else:
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
        self.cw_shapes = []
        for mtl_name, mesh in mesh_list:
            # assert(mesh.normal_indices is None)
            self.shapes.append(pyredner.Shape(
                vertices=mesh.vertices,
                indices=mesh.indices,
                material_id=material_id_map[mtl_name],
                uvs=mesh.uvs,
                normals=mesh.normals,
                uv_indices=mesh.uv_indices))

        self.camera = pyredner.automatic_camera_placement(self.shapes, resolution=(512, 512))
        # Compute the center of the teapot
        self.center = torch.mean(torch.cat(vertices), 0)
        self.translation = torch.tensor([0., 0., 0.], device=pyredner.get_device(), requires_grad=True)

        self.angle_input_adv_list = []
        self.angle_input_orig_list = []
        self.pose = pose
        if attack_type == "CW":
            self.euler_angles_modifier = torch.tensor([0., 0., 0.], device=pyredner.get_device(), requires_grad=True)
            if pose == 'forward':
                self.euler_angles = tanh_rescale(torch_arctanh(
                    torch.tensor([0., 0., 0.], device=pyredner.get_device())) + self.euler_angles_modifier)
                self.angle_input_orig_list.append(
                    tanh_rescale(torch_arctanh(torch.tensor([0., 0., 0.], device=pyredner.get_device()))))
            elif pose == 'top':
                self.euler_angles = tanh_rescale(torch_arctanh(
                    torch.tensor([0.35, 0., 0.], device=pyredner.get_device())) + self.euler_angles_modifier)
                self.angle_input_orig_list.append(
                    tanh_rescale(torch_arctanh(torch.tensor([0., 0., 0.], device=pyredner.get_device()))))
            elif pose == 'left':
                self.euler_angles = tanh_rescale(torch_arctanh(
                    torch.tensor([0., 0.50, 0.], device=pyredner.get_device())) + self.euler_angles_modifier)
                self.angle_input_orig_list.append(
                    tanh_rescale(torch_arctanh(torch.tensor([0., 0., 0.], device=pyredner.get_device()))))
            elif pose == 'right':
                self.euler_angles = tanh_rescale(torch_arctanh(
                    torch.tensor([0., -0.50, 0.], device=pyredner.get_device())) + self.euler_angles_modifier)
                self.angle_input_orig_list.append(
                    tanh_rescale(torch_arctanh(torch.tensor([0., 0., 0.], device=pyredner.get_device()))))

            self.angle_input_adv_list.append(self.euler_angles)
        else:
            if pose == 'forward':
                self.euler_angles = torch.tensor([0., 0., 0.], device=pyredner.get_device(), requires_grad=True)
            elif pose == 'top':
                self.euler_angles = torch.tensor([0.35, 0., 0.], device=pyredner.get_device(), requires_grad=True)
            elif pose == 'left':
                self.euler_angles = torch.tensor([0., 0.50, 0.], device=pyredner.get_device(), requires_grad=True)
            elif pose == 'right':
                self.euler_angles = torch.tensor([0., -0.50, 0.], device=pyredner.get_device(), requires_grad=True)

        if attack_type == "CW":
            self.light_input_orig_list = []
            self.light_input_adv_list = []
            delta = 1e-6 # constant for stability
            self.light_modifier = torch.tensor([0., 0., 0.], device=pyredner.get_device(), requires_grad=True)
            self.light_init_vals = torch.tensor([20000.0, 30000.0, 20000.0], device=pyredner.get_device())
            # redner can't accept negative light intensities, so we have to be a bit creative and work with lighting norms instead and then rescale them afterwards...
            tanh_factor = tanh_rescale(torch_arctanh(self.light_init_vals/torch.norm(self.light_init_vals)) + self.light_modifier/torch.norm(self.light_modifier + delta))
            self.light_intensity = torch.norm(self.light_init_vals) * torch.clamp(tanh_factor, 0, 1)

            self.light_input_orig_list.append(self.light_init_vals/torch.norm(self.light_init_vals))
            self.light_input_adv_list.append(self.light_intensity)
            self.light = pyredner.PointLight(
                position=(self.camera.position + torch.tensor((0.0, 0.0, 100.0))).to(pyredner.get_device()),
                intensity=self.light_intensity)
        else:
            self.light = pyredner.PointLight(
                position=(self.camera.position + torch.tensor((0.0, 0.0, 100.0))).to(pyredner.get_device()),
                intensity=Variable(torch.tensor((20000.0, 30000.0, 20000.0), device=pyredner.get_device()), requires_grad=True))

        background = pyredner.imread(background)
        self.background = background.to(pyredner.get_device())

    """
    image: the torch variable holding the image
    net_out: the output of the framework on the image
    label: an image label (given as an integer index)
    returns: the gradient of the image w.r.t the given label
    """
    def _get_gradients(self, image, net_out, label):
        score = net_out[0][label]
        score.backward(retain_graph=True)
        # return image.grad

    """
    Classifies the input image according to self.framework.
    image: np array of input image
    label: correct class label for image
    """
    def classify(self, image):
        self.framework.eval()
        # transform image before classifying by standardizing values
        mean, std = self.framework_params["mean"], self.framework_params["std"]
        normalize = transforms.Normalize(mean, std)
        image = normalize(image[0])
        image = image.unsqueeze(0)

        # forward pass
        fwd = self.framework.forward(image)

        # classification via softmax
        probs, top5 = torch.topk(fwd, 5, 1, True, True)
        top5 = top5[0]
        probs = probs[0]
        
        labels = [(self.label_names[label.item()], probs[idx].item()) for idx, label in enumerate(top5)]
        print("Top 5: ", labels)
        prediction_idx = top5[0]

        return prediction_idx, fwd


    # Model the scene based on current instance params
    def _model(self):
        # Get the rotation matrix from Euler angles
        rotation_matrix = pyredner.gen_rotate_matrix(self.euler_angles)
        self.euler_angles.retain_grad()
        # Shift the vertices to the center, apply rotation matrix,
        # shift back to the original space, then apply the translation.
        vertices = []
        if self.attack_type == "CW":
            for m, shape in zip(self.modifiers, self.shapes):
                shape_v = tanh_rescale(torch_arctanh(shape.vertices.clone().detach()) - m.clone().detach() + m)
                shape.vertices = (shape_v - self.center) @ torch.t(rotation_matrix) + self.center + self.translation
                shape.vertices.retain_grad()
                shape.vertices.register_hook(set_grad(shape.vertices))
                shape.normals = pyredner.compute_vertex_normal(shape.vertices, shape.indices)
                vertices.append(shape.vertices.clone().detach())
        else:
            for shape in self.shapes:
                shape_v = shape.vertices.clone().detach()
                shape.vertices = (shape_v - self.center) @ torch.t(rotation_matrix) + self.center + self.translation
                shape.vertices.retain_grad()
                shape.vertices.register_hook(set_grad(shape.vertices))
                shape.normals = pyredner.compute_vertex_normal(shape.vertices, shape.indices)
                vertices.append(shape.vertices.clone().detach())
        self.center = torch.mean(torch.cat(vertices), 0)
        # Assemble the 3D scene.
        scene = pyredner.Scene(camera=self.camera, shapes=self.shapes, materials=self.materials)
        # Render the scene.
        img = pyredner.render_deferred(scene, lights=[self.light], alpha=True)
        return img

    """
    Render the image properly and downsample it to the right dimensions
    out_dir = the directory you want to save the image in
    filename = the image file name
    """
    def render_image(self, out_dir=None, filename=None):
        if (out_dir is None) is not (filename is None):
            raise Exception("must provide both out dir and filename if you wish to save the image")

        dummy_img = self._model()

        # honestly dont know if this makes a difference, but...
        if self.attack_type == "CW":
            self.euler_angles_modifier.data = torch.tensor([0., 0., 0.], device=pyredner.get_device(), requires_grad=True)
            self.euler_angles = tanh_rescale(torch_arctanh(
                        torch.tensor([0., 0., 0.], device=pyredner.get_device())) + self.euler_angles_modifier)
        else:
            self.euler_angles.data = torch.tensor([0., 0., 0.], device=pyredner.get_device(),
                                                           requires_grad=True)
        img = self._model()
        # just meant to prevent rotations from being stacked onto one another with the above line

        alpha = img[:, :, 3:4]
        img = img[:, :, :3] * alpha + self.background * (1 - alpha)

        # Visualize the initial guess
        eps = 1e-6
        img = torch.pow(img + eps, 1.0 / 2.2)  # add .data to stop PyTorch from complaining
        img = torch.nn.functional.interpolate(img.T.unsqueeze(0), size=self.image_dims, mode='bilinear')
        img.retain_grad()

        # save image
        if out_dir is not None and filename is not None:
            plt.imsave(out_dir + "/" + filename, np.clip(img[0].T.data.cpu().numpy(), 0, 1))

        return img.permute(0, 1, 3, 2)

    """
    Does a random sampling attack on the image. 
    
    label: the only required parameter -- this is the index of the class you wish to decrease the network score for.
    out_dir: the directory the image should be saved in (default None; don't change if you don't wish to save the image!)
    filename: the image name of the image (e.g. "car_left.png"). Default None
    vertex_eps: the epsilon for the vertex attack. Default 0.001
    pose_eps: the epsilon for the pose attack. Default 0.05
    lighting_eps: the epsilon for the lighting attack. Default 4000 -- this is due to the intensity scale.
    vertex_attack: whether the vertex component should be attacked or not. True by default.
    pose_attack: whether the pose component should be attacked or not. True by default.
    lighting_attack: whether the lighting should be attacked or not.

    RETURNS: Prediction, 3-channel image
    """
    def attack_random_sample(self, label, out_dir=None, filename=None, vertex_eps=0.001, pose_eps=0.05, lighting_eps=4000,
                    vertex_attack=True, pose_attack=True, lighting_attack=False):
        
        if out_dir is not None and filename is None:
            raise Exception("Must provide image title if out dir is provided")
        elif filename is not None and out_dir is None:
            raise Exception("Must provide directory if image is to be saved")

        if vertex_attack:
            for shape in self.shapes:
                shape.vertices += 10 * vertex_eps * torch.rand(shape.vertices.shape) - 5 * vertex_eps

        if pose_attack:
            self.euler_angles.data += 10 * pose_eps * torch.rand(self.euler_angles.shape) - 5 * pose_eps

        if lighting_attack:
            self.light_intensity.data += 10 * lighting_eps * torch.rand(self.light_intensity.shape) - 5 * lighting_eps
        
        img = self._model()
        # just meant to prevent rotations from being stacked onto one another with the above line

        alpha = img[:, :, 3:4]
        img = img[:, :, :3] * alpha + self.background * (1 - alpha)

        # Visualize the initial guess
        eps = 1e-6
        img = torch.pow(img + eps, 1.0 / 2.2)  # add .data to stop PyTorch from complaining
        img = torch.nn.functional.interpolate(img.T.unsqueeze(0), size=self.image_dims, mode='bilinear')
        
        pred, net_out = self.classify(img)

        final_image = np.clip(img[0].permute(1, 2, 0).data.cpu().numpy(), 0, 1)
        if out_dir is not None and filename is not None:
            plt.imsave(out_dir + "/" + filename, final_image)
        
        return pred, final_image

    """
    Does an FGSM attack on the image to induce misclassification. 
    If you want to move away from a specific class, then subtract. 
    Else, if you want to move towards a specific class, then add the gradient instead.
    
    label: the only required parameter -- this is the index of the class you wish to decrease the network score for.
    out_dir: the directory the image should be saved in (default None; don't change if you don't wish to save the image!)
    filename: the image name of the image (e.g. "car_left.png"). Default None
    steps: an integer that is the number of steps you wish to perform FGSM for, Default 5.
    vertex_eps: the epsilon for the vertex FGSM attack. Default 0.001
    pose_eps: the epsilon for the pose FGSM attack. Default 0.05
    lighting_eps: the epsilon for the lighting FGSM attack. Default 4000 -- this is due to the intensity scale.
    vertex_attack: whether the vertex component should be attacked or not. True by default.
    pose_attack: whether the pose component should be attacked or not. True by default.
    lighting_attack: whether the lighting should be attacked or not.

    RETURNS: Prediction, 3-channel image
    """
    def attack_FGSM(self, label, out_dir=None, filename=None, steps=5, vertex_eps=0.001, pose_eps=0.05, lighting_eps=4000,
                    vertex_attack=True, pose_attack=True, lighting_attack=False):
        if out_dir is not None and filename is None:
            raise Exception("Must provide image title if out dir is provided")
        elif filename is not None and out_dir is None:
            raise Exception("Must provide directory if image is to be saved")

        # classify
        img = self.render_image(out_dir=out_dir, filename=filename)
        # only there to zero out gradients.
        optimizer = torch.optim.Adam([self.translation, self.euler_angles, self.light.intensity], lr=0)

        for i in range(steps):
            optimizer.zero_grad()
            pred, net_out = self.classify(img)
            if pred.item() != label and i != 0:
                print("misclassification at step ", i)
                final_image = np.clip(img[0].permute(1, 2, 0).data.cpu().numpy(), 0, 1)
                return pred, final_image
            # get gradients
            self._get_gradients(img.cpu(), net_out, label)

            delta = 1e-6
            inf_count = 0
            nan_count = 0

            # attack each shape's vertices
            if vertex_attack:
                for shape in self.shapes:
                    if not torch.isfinite(shape.vertices.grad).all():
                        inf_count += 1
                    elif torch.isnan(shape.vertices.grad).any():
                        nan_count += 1
                    else:
                        # subtract because we are trying to decrease the classification score of the label
                        shape.vertices -= torch.sign(
                            shape.vertices.grad / (torch.norm(shape.vertices.grad) + delta)) * vertex_eps

            if pose_attack:
                self.euler_angles.data -= torch.sign(
                    self.euler_angles.grad / (torch.norm(self.euler_angles.grad) + delta)) * pose_eps

            if lighting_attack:
                light_sub = torch.sign(self.light.intensity.grad / (torch.norm(self.light.intensity.grad) + delta)) * lighting_eps
                light_sub = torch.min(self.light.intensity.data, light_sub)
                self.light.intensity.data -= light_sub

            img = self.render_image(out_dir=out_dir, filename=filename)

        final_pred, net_out = self.classify(img)
        final_image = np.clip(img[0].permute(1, 2, 0).data.cpu().numpy(), 0, 1)
        return final_pred, final_image

    """
    Does a PGD attack on the image to induce misclassification. 
    If you want to move away from a specific class, then subtract. 
    Else, if you want to move towards a specific class, then add the gradient instead.
    
    label: the only required parameter -- this is the index of the class you wish to decrease the network score for.
    out_dir: the directory the image should be saved in (leave this as None if you don't wish to save the image!)
    filename: the image name of the image (e.g. "car_left.png"). Default None
    steps: an integer that is the number of steps you wish to perform PGD for. Default 5
    vertex_epsilon: the epsilon bound for the vertex PGD attack. Default 1.0
    pose_epsilon: the epsilon bound for the pose PGD attack. Default 1.0
    lighting_epsilon: the epsilon bound for the lighting PGD attack. Default 4000 -- this is due to the intensity scale.
    vertex_lr: the learning rate for the vertex PGD attack. Default 0.001
    pose_lr: the learning rate for the pose PGD attack. Default 0.05
    lighting_lr: the learning rate for the lighting PGD attack. Default 4000 -- this is due to the intensity scale.
    vertex_attack: whether the vertex component should be attacked or not. True by default.
    pose_attack: whether the pose component should be attacked or not. True by default.
    lighting_attack: whether the lighting should be attacked or not.

    RETURNS: Prediction, 3-channel image
    """
    def attack_PGD(self, label, out_dir=None, filename=None, steps=5, vertex_epsilon=1.0, pose_epsilon=1.0, lighting_epsilon=8000.0,
                   vertex_lr=0.001, pose_lr=0.05, lighting_lr=4000.0,
                   vertex_attack=True, pose_attack=True, lighting_attack=False):

        if out_dir is not None and filename is None:
            raise Exception("Must provide image title if out dir is provided")
        elif filename is not None and out_dir is None:
            raise Exception("Must provide directory if image is to be saved")

        # classify
        img = self.render_image(out_dir=out_dir, filename=filename)

        # only there to zero out gradients.
        optimizer = torch.optim.Adam([self.translation, self.euler_angles, self.light.intensity], lr=0)

        for i in range(steps):
            optimizer.zero_grad()
            pred, net_out = self.classify(img)
            if pred.item() != label and i != 0:
                print("misclassification at step ", i)
                final_image = np.clip(img[0].permute(1, 2, 0).data.cpu().numpy(), 0, 1)
                return pred, final_image
            # get gradients
            self._get_gradients(img.cpu(), net_out, label)
            delta = 1e-6
            inf_count = 0
            nan_count = 0

            if vertex_attack:
                # attack each shape's vertices
                for shape in self.shapes:
                    if not torch.isfinite(shape.vertices.grad).all():
                        inf_count += 1
                    elif torch.isnan(shape.vertices.grad).any():
                        nan_count += 1
                    else:
                        # subtract because we are trying to decrease the classification score of the label
                        shape.vertices -= torch.clamp(
                            shape.vertices.grad / (torch.norm(shape.vertices.grad) + delta) * vertex_lr,
                            -vertex_epsilon, vertex_epsilon)

            if lighting_attack:
                light_sub = torch.clamp(
                    self.light.intensity.grad / (torch.norm(self.light.intensity.grad) + delta) * lighting_lr, -lighting_epsilon,
                    lighting_epsilon)
                light_sub = torch.min(self.light.intensity.data, light_sub)
                self.light.intensity.data -= light_sub

            if pose_attack:
                self.euler_angles.data -= torch.clamp(
                    self.euler_angles.grad / (torch.norm(self.euler_angles.grad) + delta) * pose_lr, -pose_epsilon,
                    pose_epsilon)

            img = self.render_image(out_dir=out_dir, filename=filename)

        final_pred, net_out = self.classify(img)
        final_image = np.clip(img[0].permute(1, 2, 0).data.cpu().numpy(), 0, 1)
        return final_pred, final_image

    """
    The Carlini-Wagner loss.
    """
    def cw_loss(self, output, target, dist, scale_const):
        # compute the probability of the label class versus the maximum other
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        if self.targeted:
            # if targeted, optimize for making the other class most likely
            loss1 = torch.clamp(other - real, min=0.)  # equiv to max(..., 0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            loss1 = torch.clamp(real - other, min=0.)  # equiv to max(..., 0.)
        loss1 = torch.sum(scale_const * loss1)

        loss2 = dist.sum()
        # print(loss1, loss2)
        loss = loss1 + loss2
        return loss

    """
    Does a Carlini-Wagner attack on the image to induce misclassification. 
    If you want to move away from a specific class, then subtract. 
    Else, if you want to move towards a specific class, then add the gradient instead.
    
    label: the only required parameter -- this is the index of the class you wish to decrease the network score for.
    out_dir: the directory the image should be saved in (default None; don't change if you don't wish to save the image!)
    filename: the image name of the image (e.g. "car_left.png"). Default None
    steps: an integer that is the number of steps you wish to perform CW for, Default 5.
    vertex_lr: the epsilon for the vertex CW attack. Default 0.001
    pose_lr: the epsilon for the pose CW attack. Default 0.05
    lighting_eps: the epsilon for the lighting CW attack. Default 8000 -- this is due to the intensity scale.
    vertex_attack: whether the vertex component should be attacked or not. True by default.
    pose_attack: whether the pose component should be attacked or not. True by default.
    lighting_attack: whether the lighting should be attacked or not.
    target: the target label. Default None.

    RETURNS: Prediction, 3-channel image
    """
    def attack_cw(self, label, out_dir=None, filename=None, steps=5,
                  vertex_lr=0.001, pose_lr=0.05, lighting_lr=8000,
                  vertex_attack=True, pose_attack=True, lighting_attack=False, target=None):

        if out_dir is not None and filename is None:
            raise Exception("Must provide image title if out dir is provided")
        elif filename is not None and out_dir is None:
            raise Exception("Must provide directory if image is to be saved")

        # classify
        img = self.render_image(out_dir=out_dir, filename=filename)

        if target is not None:
            target = torch.tensor([target]).to(pyredner.get_device())
            self.targeted = True
        else:
            target = torch.tensor([label]).to(pyredner.get_device())

        target_onehot = torch.zeros(target.size() + (self.NUM_CLASSES,)).to(pyredner.get_device())
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)

        # only there to zero out gradients.
        optimizer = torch.optim.Adam([self.translation, self.euler_angles_modifier, self.light_modifier] + [m for m in self.modifiers], lr=0)

        for i in range(steps):
            optimizer.zero_grad()
            pred, net_out = self.classify(img)
            if pred.item() != label and i != 0:
                final_image = np.clip(img[0].permute(1, 2, 0).data.cpu().numpy(), 0, 1)
                return pred, final_image

            loss = 0
            if vertex_attack:
                dist = l2_dist(self.input_adv_list, self.input_orig_list, False)
                loss += self.cw_loss(net_out, target_onehot, dist, 0.1)

            if pose_attack:
                dist = l2_dist(self.angle_input_adv_list, self.angle_input_orig_list, False)
                loss += self.cw_loss(net_out, target_onehot, dist, 0.1)

            if lighting_attack:
                dist = l2_dist(self.light_input_adv_list, self.light_input_orig_list, False)
                loss += self.cw_loss(net_out, target_onehot, dist, 0.1)

            # get gradients
            loss.backward(retain_graph=True)

            delta = 1e-6
            inf_count = 0
            nan_count = 0

            if vertex_attack:
                # attack each shape's vertices
                self.input_orig_list = []
                self.input_adv_list = []

                for shape, m in zip(self.shapes, self.modifiers):
                    shape.vertices = tanh_rescale(torch_arctanh(shape.vertices.clone().detach()) - m.clone().detach())
                    if not torch.isfinite(m.grad).all():
                        inf_count += 1
                    elif torch.isnan(m.grad).any():
                        nan_count += 1
                    else:
                        # subtract because we are trying to decrease the classification score of the label
                        m.data -= m.grad / (
                            torch.norm(m.grad) + delta) * vertex_lr


                for shape, m in zip(self.shapes, self.modifiers):
                    self.input_orig_list.append(tanh_rescale(torch_arctanh(shape.vertices)))
                    shape.vertices = tanh_rescale(torch_arctanh(shape.vertices) + m)

                    self.input_adv_list.append(shape.vertices)

            if lighting_attack:
                self.light_input_orig_list = []
                self.light_input_adv_list = []
                # tanh_rescale(torch_arctanh(self.light_init_vals/torch.norm(self.light_init_vals)) + self.light_modifier/torch.norm(self.light_modifier + delta))
                tanh_factor = tanh_rescale(torch_arctanh(self.light_intensity.clone().detach()/torch.norm(self.light_intensity.clone().detach())) 
                                - self.light_modifier.clone().detach()/torch.norm(self.light_modifier.clone().detach() + delta))
                self.light_init_vals = torch.norm(self.light_intensity.clone().detach()) * torch.clamp(tanh_factor, 0, 1)

                self.light_modifier.data -= self.light_modifier.grad / (torch.norm(self.light_modifier.grad) + delta) * lighting_lr

                # redner can't accept negative light intensities, so we have to be a bit creative and work with lighting norms instead and then rescale them afterwards...
                tanh_factor = tanh_rescale(torch_arctanh(self.light_init_vals/torch.norm(self.light_init_vals)) + self.light_modifier/torch.norm(self.light_modifier + delta))
                self.light_intensity = torch.norm(self.light_init_vals) * torch.clamp(tanh_factor, 0, 1)

                self.light_input_orig_list.append(self.light_init_vals/torch.norm(self.light_init_vals))
            
                self.light_input_adv_list.append(self.light_intensity)
                self.light = pyredner.PointLight(
                    position=(self.camera.position + torch.tensor((0.0, 0.0, 100.0))).to(pyredner.get_device()),
                    intensity=self.light_intensity)

            if pose_attack:
                self.angle_input_adv_list = []
                self.angle_input_orig_list = []

                self.euler_angles_modifier.data -= self.euler_angles_modifier.grad / (
                            torch.norm(self.euler_angles_modifier.grad) + delta) * pose_lr
                self.euler_angles = tanh_rescale(torch_arctanh(torch.tensor([0., 0., 0.], device=pyredner.get_device())) + self.euler_angles_modifier)
                self.angle_input_orig_list.append(tanh_rescale(torch_arctanh(torch.tensor([0., 0., 0.], device=pyredner.get_device()))))
                self.angle_input_adv_list.append(self.euler_angles)

            img = self.render_image(out_dir = out_dir, filename=filename)

        final_pred, net_out = self.classify(img)
        final_image = np.clip(img[0].permute(1, 2, 0).data.cpu().numpy(), 0, 1)
        return final_pred, final_image
