from .EndiannessCount import *
from .GrayScaleImage import *

from .Vector import *


def get_transform(**kwargs):

    transform_name = kwargs["name"]

    if transform_name == "EndiannessCount":
        return EndiannessCount()

    elif transform_name == "GrayScaleImage":
        return GrayScaleImage(
            dimx=kwargs["dimx"],
            dimy=kwargs["dimy"],
            normalize=kwargs["normalize"],
            duplicate_to_n_channels=kwargs["duplicate_to_n_channels"],
        )

    elif transform_name == "Vector1D":
        return Vector1D(length=kwargs["length"])

    raise ValueError(f"Unknown transform: {transform_name}")
