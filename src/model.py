"""Models for training."""

import torch.nn as nn
from torchvision import models
from torchvision.models import Inception_V3_Weights


def get_model(): 
    model_inception_v3 = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model_inception_v3.aux_logits = False
    # num_features - размерность вектора фич, поступающего на вход FC
    num_features = 2048
    # n_classes - количество классов, которые будет предсказывать наша модель
    n_classes = 2
    # Заменяем Fully-Connected слой на наш линейный классификатор
    model_inception_v3.fc = nn.Linear(in_features=num_features, out_features=n_classes)
    model_inception_v3.AuxLogits.fc = nn.Linear(768, 2)

    return model_inception_v3