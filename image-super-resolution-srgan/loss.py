import torch.nn as nn
from torchvision.models import vgg19
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(DEVICE)
        self.loss = nn.MSELoss()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input = self.vgg(input)
        vgg_target = self.vgg(target)
        return self.loss(vgg_input, vgg_target)
    
