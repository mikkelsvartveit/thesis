from .EmbeddingModels import *
from .SimpleCnnModels import *
from .EmbeddingResNet import ResNet50Embedding
from torchvision.models import resnet50


def get_model(**kwargs):
    name = kwargs["name"]
    if name == "Simple1dEmbedding":
        return Simple1dEmbedding
    elif name == "Simple2dEmbedding":
        return Simple2dEmbedding
    elif name == "Simple1d":
        return Simple1d
    elif name == "Simple2d":
        return Simple2d
    elif name == "ResNet50":
        return resnet50
    elif name == "ResNet50Embedding":
        return ResNet50Embedding
    else:
        raise ValueError(f"Model {name} not found")
