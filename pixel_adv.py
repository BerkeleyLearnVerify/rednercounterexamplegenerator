import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models.vgg as vgg
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

NUM_CLASSES = 12
vgg16 = vgg.vgg16(pretrained=True)
num_ftrs = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)

if not torch.cuda.is_available():
    vgg16.load_state_dict(torch.load('torch_models/model_ft.pt', map_location=lambda storage, location: storage))
else:
    vgg16.load_state_dict(torch.load('torch_models/model_ft.pt'))    

class PixelPerturb:
    def __init__(self, framework, framework_shape, normalize_params):
        self.framework = framework
        self.framework_shape = framework_shape
        self.framework_params = normalize_params
        
    def _get_gradients(net_out, label):
        score = net_out[0][label]
        score.backward(retain_graph=True)

    def classify(self, img):
        self.framework.eval()
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
        pred = top5[0]
        return fwd, pred

    def FGSM(self, img, label, eps=0.01, iters=5, save=False, save_path=None):
        assert img.shape == self.framework_shape
        img = torch.tensor(img)
        img = img.permute(0, 1, 3, 2)
        img = Variable(img, requires_grad=True)

        optimizer = torch.optim.Adam([img], lr=0)
        for i in range(iters):
            net_out, pred = classify(img)
            if pred.item() != label and i != 0:
                img_return = img[0].permute(1, 2, 0).cpu().data
                return img, pred.item()
            self._get_gradients(net_out, label)
            img -= eps * torch.sign(img.grad/torch.norm(img.grad))
            optimizer.zero_grad()
        img_return = img[0].permute(1, 2, 0).cpu().data
        return img.cpu().data, pred.item()

    def PGD(self, img, label, lr=0.01, epsilon=0.5, iters=5, save=False, save_path=None):
        assert img.shape == self.framework_shape
        img = torch.tensor(img)
        img = img.permute(0, 1, 3, 2)
        img = Variable(img, requires_grad=True)

        optimizer = torch.optim.Adam([img], lr=0)
        for i in range(iters):
            net_out, pred = classify(img)
            if pred.item() != label and i != 0:
                img_return = img[0].permute(1, 2, 0).cpu().data
                return img, pred.item()
            self._get_gradients(net_out, label)
            img -= lr * torch.clamp(img.grad/torch.norm(img.grad), -epsilon, epsilon)
            optimizer.zero_grad()

        img_return = img[0].permute(1, 2, 0).cpu().data
        return img_return, pred.item()

    def _reduce_sum(x, keepdim=True):
        # silly PyTorch, when will you get proper reducing sums/means?
        for a in reversed(range(1, x.dim())):
            x = x.sum(a, keepdim=keepdim)

        return x


    def _l2_dist(x, y, keepdim=True):
        d = None
        for x_i, y_i in zip(x, y):
            if d is not None:
                d += torch.sum(reduce_sum((x_i - y_i) ** 2, keepdim=keepdim))
            else:
                d = torch.sum(reduce_sum((x_i - y_i) ** 2, keepdim=keepdim))

        return d


    def _torch_arctanh(x, eps=1e-6):
        x *= (1. - eps)
        return (torch.log((1 + x) / (1 - x))) * 0.5


    def _tanh_rescale(x, x_min=-1., x_max=1.):
        return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min

    def _CW(self, img, label, target=None, lr=0.05):
        img = torch.tensor(img)
        img = img.permute(0, 1, 3, 2)
        img = Variable(img, requires_grad=True)

        orig_img = img
        img_modifier = torch.zeros(img.size(), device=pyredner.get_device(), requires_grad=True)
        img = tanh_rescale(torch_arctanh(img) + self.img_modifier)
        modified_img = img
        
        if target is not None:
            target = torch.tensor([target]).to(pyredner.get_device())
            self.targeted = True
        else:
            target = torch.tensor([label]).to(pyredner.get_device())

        target_onehot = torch.zeros(target.size() + (NUM_CLASSES,)).to(pyredner.get_device())
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        optimizer = torch.optim.Adam([img, img_modifier])
        for i in range(steps):
            optimizer.zero_grad()
            pred, net_out = self.classify(img)
            if pred.item() != label and i != 0:
                img_return = img[0].permute(1, 2, 0).cpu().data
                return pred, img_return

            loss = 0
            dist = l2_dist(input_adv_list, input_orig_list, False)
            loss += self.cw_loss(net_out, target_onehot, dist, 0.1)

            loss.backward(retain_graph=True)

            delta = 1e-6
            inf_count = 0
            nan_count = 0

            # attack each shape's vertices
            self.input_orig_list = []
            self.input_adv_list = []

            if not torch.isfinite(self.img_modifier.grad).all():
                inf_count += 1
            elif torch.isnan(self.img_modifier.grad).any():
                nan_count += 1
            else:
                # subtract because we are trying to decrease the classification score of the label
                old_modifier = img_modifier.clone().detach()
                img_modifier.data -= (img_modifier.grad / (
                    torch.norm(img_modifier.grad) + delta)) * lr

                img = tanh_rescale(torch_arctanh(orig_img) - old_modifier + self.img_modifier)

        final_pred, net_out = self.classify(img)
        img_return = img[0].permute(1, 2, 0).cpu().data
        return final_pred, img_return