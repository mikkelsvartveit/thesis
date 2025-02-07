from .EndiannessModels import *
from .MINOS_cnn import *
from .EmbeddingModels import *


def get_model(**kwargs):
    name = kwargs["name"]
    if name == "MINOS_cnn":
        return MINOS
    elif name == "EndiannessModel":
        return EndiannessModel
    elif name == "EmbeddingModel":
        return EmbeddingAndCNNModel
    else:
        raise ValueError(f"Model {name} not found")
