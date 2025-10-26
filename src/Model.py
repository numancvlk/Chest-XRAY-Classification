#LIBRARIES
import torch
from torch import nn
from torchvision import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def getModel(numClasses = 2, device = "cpu"):
    
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
        param.requires_grad = False

    numFeatures = model.classifier.in_features 

    model.classifier = nn.Linear(in_features=numFeatures,
                                 out_features=numClasses)
    
    model.to(device)
    
    return model