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
        img = torch.Variable(img, requires_grad=True)
        optimizer = torch.optim.Adam([img], lr=0)
        for i in range(iters):
            net_out, pred = classify(img)
            if pred.item() != label and i != 0:
                return img.cpu().data, pred.item()
            self._get_gradients(net_out, label)
            img -= eps * torch.sign(img.grad/torch.norm(img.grad))
            optimizer.zero_grad()

        return img.cpu().data, pred.item()

    def PGD(self, img, label, lr=0.01, epsilon=0.5, iters=5, save=False, save_path=None):
        assert img.shape == self.framework_shape
        img = torch.Variable(img, requires_grad=True)
        optimizer = torch.optim.Adam([img], lr=0)
        for i in range(iters):
            net_out, pred = classify(img)
            if pred.item() != label and i != 0:
                return img.cpu().data, pred.item()
            self._get_gradients(net_out, label)
            img -= lr * torch.clamp(img.grad/torch.norm(img.grad), -epsilon, epsilon)
            optimizer.zero_grad()
        return img.cpu().data, pred.item()

    def CW(self, img, label):
        