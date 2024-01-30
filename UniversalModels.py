# UniversalModels.py 2023
import torch

from typing import Callable

class UniversalFunction(torch.nn.Module):

    def __init__(self, f: Callable[[torch.tensor], torch.tensor], theta:torch.tensor):
        
        super().__init__()
        # make weights torch parameters
        self.weights = torch.nn.Parameter(theta)
        self.f = f        
        
    def forward(self):
        return self.f(self.weights)
        # return preds

class UniversalFunctionWithData(torch.nn.Module):

    def __init__(self, f: Callable[[torch.tensor, torch.tensor], torch.tensor], theta:torch.tensor):
        
        super().__init__()
        # make weights torch parameters
        self.weights = torch.nn.Parameter(theta)
        self.f = f        
        
    def forward(self, X):
        return self.f(self.weights, X)
        # return preds

# EoF.
