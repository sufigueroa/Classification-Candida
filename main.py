import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from utils import *

PATH = 'images/Low concentration 1.tiff'
# PATH = 'images/Medium concentration 1.tiff'
# PATH = 'images/High concentration 1.tiff'

# Leer tiff y convertirlo en np.array
im = read_tiff(PATH)

segmentate_matrix(im)