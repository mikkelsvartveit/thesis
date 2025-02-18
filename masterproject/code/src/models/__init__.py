from .EndiannessModels import *
from .MINOS_cnn import *
from .EmbeddingModels import *
from .SimpleCnnModels import *


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
    elif name == "Simple1DCNN":
        return Simple1DCNN
    elif name == "Simple2dCNN":
        return Simple2dCNN
    else:
        raise ValueError(f"Model {name} not found")
