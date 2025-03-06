from .EndiannessModels import *
from .MINOS_cnn import *
from .EmbeddingModels import *
from .SimpleCnnModels import *
from .EmbeddingResNet import EmbeddingResNet50
from torchvision.models import resnet50


def get_model(**kwargs):
    name = kwargs["name"]
    if name == "MINOS_cnn":
        return MINOS
    elif name == "EndiannessModel":
        return EndiannessModel
    elif name == "EmbeddingAndCNNModel":
        return EmbeddingAndCNNModel
    elif name == "EmbeddingAnd2dCNNModel":
        return EmbeddingAnd2dCNNModel
    elif name == "Simple1dCNN":
        return Simple1dCNN
    elif name == "Simple2dCNN":
        return Simple2dCNN
    elif name == "ResNet50":
        return resnet50
    elif name == "EmbeddingResNet50":
        return EmbeddingResNet50
    else:
        raise ValueError(f"Model {name} not found")
