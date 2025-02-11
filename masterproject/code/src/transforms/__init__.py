from .EndiannessCount import *
from .GrayScaleImage import *

from .Vector import *


def get_transform(**kwargs):

    transform_name = kwargs["name"]

    if transform_name == "EndiannessCount":
        return EndiannessCount()

    elif transform_name == "GrayScaleImage":
        return GrayScaleImage(
            dimx=kwargs["params"]["dimx"],
            dimy=kwargs["params"]["dimy"],
            normalize=kwargs["params"]["normalize"],
            duplicate_to_n_channels=kwargs["params"]["duplicate_to_n_channels"],
        )

    elif transform_name == "Vector1D":
        return Vector1D(length=kwargs["params"]["length"])

    raise ValueError(f"Unknown transform: {transform_name}")
