import torch
import torch.nn as nn


class ClassificationAccuracy(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, prob, label):
        _, pred = torch.max(prob, dim=1)
        correct = (pred == label).sum().item()
        acc = correct / label.size(0)
        return acc
