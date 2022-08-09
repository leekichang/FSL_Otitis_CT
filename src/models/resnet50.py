import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Sim_ResNet50():
    def __init__(self, DEVICE='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE)
        for param in self.model.parameters():
            param.requires_grad = False
        self.sim_func = nn.Linear(in_features = 100352*2 , out_features = 1).to(DEVICE)
        self.DEVICE = DEVICE

    def pair_forward(self, X):
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

class Cosine_ResNet50():
    def __init__(self, DEVICE='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE)
        for param in self.model.parameters():
            param.requires_grad = False
        self.embedding_layer = nn.Linear(in_features = 2048 , out_features = 512).to(DEVICE)
        self.DEVICE = DEVICE
        self.cosine_sim = torch.nn.CosineSimilarity()

    def pair_forward(self, X):
        bs = X.shape[0]
        imgs1, imgs2 = X[:, 0], X[:, 1]
        features1 = torch.from_numpy(np.array([])).float().to(self.DEVICE)
        features2 = torch.from_numpy(np.array([])).float().to(self.DEVICE)
        for idx in range(bs):
            features1 = torch.cat((features1, self.custom_forward(imgs1[idx]).flatten(0, 2).reshape(1, -1)))
            features2 = torch.cat((features2, self.custom_forward(imgs2[idx]).flatten(0, 2).reshape(1, -1)))
        output = self.cosine_sim(features1, features2)
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
        out = out.squeeze(-1).permute(0, 2, 1)
        out = self.embedding_layer(out)
        return out

if __name__ == '__main__':
    sim_resnet = Cosine_ResNet50()
    x = torch.randn((2, 2, 49, 3, 224, 224)).to(sim_resnet.DEVICE)
    output = sim_resnet.pair_forward(x)
    print(output)
    