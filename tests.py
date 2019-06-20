import bfseg
from skimage.io import imread
import numpy as np

d = imread("data/test.tif")


def test_find_focus():
    assert bfseg.find_focus(d) == 17
