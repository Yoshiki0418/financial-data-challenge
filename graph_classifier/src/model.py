import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50BinaryClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # 最終全結合層を置き換え（二値分類のため）
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)  
        
        # シグモイド関数を用いて確率値を出力
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x) 
        return x
