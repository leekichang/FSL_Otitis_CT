import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Sim_ResNet50():
    def __init__(self, DEVICE='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE)
        for param in self.model.parameters():
            param.requires_grad = False
        self.sim_func = sim_func = nn.Linear(in_features = 100352*2 , out_features = 1).to(DEVICE)
        self.DEVICE = DEVICE

    def simmilarity_forward(self, X):
        bs = X.shape[0]
        imgs1, imgs2 = X[:, 0], X[:, 1]
        output = torch.from_numpy(np.array([])).float().to(self.DEVICE)
        for idx in range(bs):
            feature = torch.cat((self.custom_forward(imgs1[idx]).flatten(0, 2), self.custom_forward(imgs2[idx]).flatten(0, 2)))
            output = torch.cat((output, feature.transpose(0, 1)), 0)
        output = self.sim_func(output)
        return output
        
        
    def custom_forward(self, X):
        out = self.model.conv1(X)
        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        out = self.model.avgpool(out)
        return out

if __name__ == '__main__':
    sim_resnet = Sim_ResNet50()
    x = torch.randn((2, 2, 49, 3, 224, 224)).to(sim_resnet.DEVICE)
    output = sim_resnet.simmilarity_forward(x)
    print(output.shape)
    
    #model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    #sim_func = nn.Linear(in_features = 100352*2 , out_features = 1).to('cuda' if torch.cuda.is_available() else 'cpu')
    