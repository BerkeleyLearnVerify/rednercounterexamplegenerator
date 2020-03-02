import torch
import torchvision
import torchvision.models.vgg as vgg
from torch.autograd import Variable
import pyredner
import matplotlib.pyplot as plt
import urllib
import zipfile

vgg16 = vgg.vgg16(pretrained=True)
def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook


class SemanticPerturbations:
    def __init__(self, framework, filename):
        self.framework = framework

        self.objects = pyredner.load_obj(filename, return_objects=True)
        self.camera = pyredner.automatic_camera_placement(self.objects, resolution=(512,512))
        vertices = []
        for obj in self.objects:
            vertices.append(obj.vertices)
            obj.vertices = Variable(obj.vertices, requires_grad=True)
            obj.vertices.retain_grad()
        
        # Compute the center of the teapot
        self.center = torch.mean(torch.cat(vertices), 0)
        self.translation = torch.tensor([10.0, -10.0, 10.0], device = pyredner.get_device(), requires_grad=True)
        self.euler_angles = torch.tensor([0.1, -0.1, 0.1], device = pyredner.get_device(), requires_grad=True)

    # image: the torch variable holding the image
    # net_out: the output of the framework on the image
    # label: the label of the image
    # returns: the gradient of the image w.r.t correct label
    def _get_gradients(self, image, net_out, label):
        score = net_out[0][label]
        score.backward(retain_graph=True)
        #return image.grad

    # classifies the input image 
    # image: np array of input image
    # label: correct class label for image
    # gradients: flag indicating whether to return grad of image
    # with respect to correct class label
    def classify(self, image, label):
        self.framework.eval()
        fwd = self.framework.forward(image.cpu())
        prediction = torch.argmax(fwd[0])
        print(prediction)
        return prediction, fwd
    # You might need to combine the detector and the renderer into one class...this will enable you to retrieve gradients of the placement w.r.t the input stuff

    # model the scene based on current instance params
    def _model(self):
        # Get the rotation matrix from Euler angles
        rotation_matrix = Variable(pyredner.gen_rotate_matrix(self.euler_angles), requires_grad=True)
        rotation_matrix.retain_grad()
        # Shift the vertices to the center, apply rotation matrix,
        # shift back to the original space, then apply the translation.
        for obj in self.objects:
            obj.vertices = (obj.vertices - self.center) @ torch.t(rotation_matrix) + self.center + self.translation
            obj.vertices.retain_grad()
        # Assemble the 3D scene.
        scene = pyredner.Scene(camera = self.camera, objects = self.objects)
        # Render the scene.
        img = pyredner.render_albedo(scene)
        img.retain_grad()
        return img

    # render the image properly and downsample it to the right dimensions
    def render_image(self):
        img = self._model()
        # Visualize the initial guess
        eps = 1e-6
        img = torch.pow(img + eps, 1.0/2.2) # add .data to stop PyTorch from complaining
        img = torch.nn.functional.interpolate(img.T.unsqueeze(0), size=(224,224), mode='bilinear')
        img.retain_grad()
        return img

    def attack(self):
        # classify 
        learning_rate = 1e-4
        img = self.render_image()
        plt.imsave("out_images/base.png", img[0].T.data.cpu().numpy())
        for i in range(10):
            pred, net_out = self.classify(img, 5)
            # get gradients
            self._get_gradients(img.cpu(), net_out, pred)
            for obj in self.objects:
                obj.vertices += obj.vertices.grad/torch.norm(obj.vertices.grad) * learning_rate
            if self.translation.grad is None:
                print("No translation grad")
            else:
                print("Hey!")
                self.translation = self.translation + self.translation.grad/torch.norm(self.translation.grad) * learning_rate
                self.translation.retain_grad()
            img = self.render_image()
            plt.imsave("out_images/img_test_" + str(i) + ".png", img[0].T.data.cpu().numpy())
        final_pred, net_out = self.classify(img, 5)
        print(final_pred)
        #plt.imsave("img_test.png", img[0].T.data.cpu().numpy())

v = SemanticPerturbations(vgg16, "teapot/teapot.obj")
v.attack()
