import os
import torch
import torch.nn as nn


class UnimodalClassifier(nn.Module):
    """
    Clasificador unimodal genérico. 
    Toma el adaptador correspondiente a la modalidad (que proyecta a 512 dimensiones)
    y le añade capas lineales + BatchNorm + dropout + ReLU, devolviendo el logit en bruto para BCEWithLogitsLoss
    """
    def __init__(self, adapter, proj_dim = 512, hidden_mlp=128, dropout_prob=0.5):
        super(UnimodalClassifier, self).__init__()

        # 1. Adaptador
        self.adapter = adapter
        
        # 2. Clasificador MLP
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_mlp, 1) #Salida LOGIT
        )
        
    def forward(self, x):
        features = self.adapter(x)
        output = self.classifier(features)
        return output