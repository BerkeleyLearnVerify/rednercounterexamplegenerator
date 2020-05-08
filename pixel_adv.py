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
        
    def _get_gradients(self, net_out, label):
        score = net_out[0][label]
        score.backward()

    def classify(self, img):
        self.framework.eval()
        mean, std = self.framework_params["mean"], self.framework_params["std"]
        normalize = transforms.Normalize(mean, std)
        image = normalize(img[0])
        image = image.unsqueeze(0)
        # forward pass
        fwd = self.framework.forward(img)
        # classification via softmax
        probs, top5 = torch.topk(fwd, 5, 1, True, True)
        top5 = top5[0]
        probs = probs[0]
        pred = top5[0]
        print("Prediction ", pred, "Probs ", probs)
        return fwd, pred

    def FGSM(self, img, label, eps=0.000, iters=5, save=False, save_path=None):
        img = torch.nn.functional.interpolate(torch.tensor(img).T.unsqueeze(0), size=self.framework_shape, mode='bilinear')
        # img = torch.tensor(img)
        img = img.permute(0, 1, 3, 2)
        img = Variable(img, requires_grad=True)
        img.retain_grad()
        optimizer = torch.optim.Adam([img], lr=0)
        for i in range(iters):
            net_out, pred = self.classify(img.clone())
            if pred.item() != label and i != 0:
                img_return = img[0].permute(1, 2, 0).cpu().data.numpy()
                return pred.item(), img_return
            self._get_gradients(net_out, label)
            img.data -= eps * torch.sign(img.grad)
            optimizer.zero_grad()
        img_return = img[0].permute(1, 2, 0).cpu().data.numpy()
        return pred.item(), img_return

    def PGD(self, img, label, lr=0.01, epsilon=0.5, iters=5, save=False, save_path=None):
        img = torch.nn.functional.interpolate(torch.tensor(img).T.unsqueeze(0), size=self.framework_shape, mode='bilinear')
        # img = torch.tensor(img)
        img = img.permute(0, 1, 3, 2)
        img = Variable(img, requires_grad=True)
        img.retain_grad()
        optimizer = torch.optim.Adam([img], lr=0)
        for i in range(iters):
            net_out, pred = self.classify(img.clone())
            if pred.item() != label and i != 0:
                img_return = img[0].permute(1, 2, 0).cpu().data.numpy()
                return pred.item(), img_return
            self._get_gradients(net_out, label)
            img.data -= torch.clamp(lr * img.grad, -epsilon, epsilon)
            optimizer.zero_grad()

        img_return = img[0].permute(1, 2, 0).cpu().data.numpy()
        return pred.item(), img_return

    def _reduce_sum(self, x, keepdim=True):
        # silly PyTorch, when will you get proper reducing sums/means?
        for a in reversed(range(1, x.dim())):
            x = x.sum(a, keepdim=keepdim)

        return x


    def _l2_dist(self, x, y, keepdim=True):
        d = None
        for x_i, y_i in zip(x, y):
            if d is not None:
                d += torch.sum(self._reduce_sum((x_i - y_i) ** 2, keepdim=keepdim))
            else:
                d = torch.sum(self._reduce_sum((x_i - y_i) ** 2, keepdim=keepdim))

        return d


    def _torch_arctanh(self, x, eps=1e-6):
        x.data *= (1. - eps)
        return (torch.log((1 + x) / (1 - x))) * 0.5


    def _tanh_rescale(self, x, x_min=-1., x_max=1.):
        return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min

    def _cw_loss(self, output, target, dist, scale_const, targeted):
        # compute the probability of the label class versus the maximum other
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        if targeted:
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

    def CW(self, img, label, iters=5, target=None, lr=0.05):
        img = torch.nn.functional.interpolate(torch.tensor(img).T.unsqueeze(0), size=self.framework_shape, mode='bilinear')
        img = img.permute(0, 1, 3, 2)
        orig_img = Variable(img, requires_grad=True)
        orig_img.retain_grad()

        orig_img = img
        img_modifier = Variable(torch.zeros(img.size()), requires_grad=True)
        img = self._tanh_rescale(self._torch_arctanh(orig_img) + img_modifier)
        modified_img = img
        
        if target is not None:
            target = torch.tensor([target])
            targeted = True
        else:
            target = torch.tensor([label])
            targeted = False


        target_onehot = torch.zeros(target.size() + (NUM_CLASSES,))
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        optimizer = torch.optim.Adam([orig_img, img_modifier], lr=0)

        for i in range(iters):
            optimizer.zero_grad()
            net_out, pred = self.classify(img.clone())
            if pred.item() != label and i != 0:
                img_return = img[0].permute(1, 2, 0).cpu().data.numpy()
                return pred, img_return

            loss = 0
            dist = self._l2_dist(orig_img, modified_img, False)
            loss += self._cw_loss(net_out, target_onehot, dist, 0.1, targeted)

            loss.backward(retain_graph=True)

            delta = 1e-6
            inf_count = 0
            nan_count = 0
            old_modifier = img_modifier.clone().detach()
            img_modifier.data -= (img_modifier.grad/torch.norm(img_modifier.grad)) * lr
            # print(img_modifier)
            img = self._tanh_rescale(self._torch_arctanh(orig_img) - old_modifier + img_modifier)
            modified_img = img

        net_out, pred = self.classify(img)
        img_return = img[0].permute(1, 2, 0).cpu().data.numpy()
        return pred.item(), img_return
