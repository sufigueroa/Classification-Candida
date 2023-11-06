from numpy import cos, sin
import numpy as np
import math
from PIL import Image
import cv2
from skspatial.objects import Line, Plane, Vector
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import label, regionprops
from skimage.color import label2rgb


PATH = 'results/'
###################################################
### Funciones Generales
###################################################

# https://stackoverflow.com/questions/18602525/python-pil-for-loop-to-work-with-multi-image-tiff
# Funcion para abrir la imagen tiff y dejarla como una matriz
def read_tiff(path):
    """
    path - Path to the multipage-tiff file
    """
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)

def save_image(img, path):
    img = Image.fromarray(np.uint8(img), 'L')
    img.save(path)

def save_rgb_image(img, path):
    img = Image.fromarray(np.uint8(img), 'RGB')
    img.save(path)

def save_video(frames, path):
    video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), 10, (frames[0].shape[1], frames[0].shape[0]), 0)
    for frame in frames:
        video.write(np.uint8(frame))
    video.release()

###################################################


###################################################
### Funciones de Interpolacion
###################################################

def isotropic_interpolation(matrix):
    new = []
    for i in range(len(matrix) - 1):
        img1 = matrix[i]
        img2 = matrix[i + 1]
        ab = np.dstack((img1, img2))
        inter = np.mean(ab, axis=2, dtype=ab.dtype) 
        new.append(img1)
        new.append(inter)
    new.append(img2)
    return np.array(new)

###################################################



def equalize(matrix):
    max_value = np.max(matrix)
    return matrix / max_value * 255

def initial_segmentation(img, umbral):
    segmented = (img > umbral) * 255
    return segmented.astype(np.uint8)

def denoise(img):
    k_first_erosion            = np.ones((3,3), np.uint8) 
    k_first_dilation           = np.ones((2,2), np.uint8) 
    k_second_dilation          = np.ones((15,15), np.uint8) 
    k_second_erosion           = np.ones((9,9), np.uint8) 

    img_dilation = cv2.dilate(img, k_first_dilation) 
    img_erosion = cv2.erode(img, k_first_erosion)
    img_dilation = cv2.dilate(img_erosion, k_second_dilation) 
    img_erosion = cv2.erode(img_dilation, k_second_erosion)
    return img_erosion

def get_mask(index, matrix, OR=True):
    if OR:
        neigh = []
        offset = 8
        indexes = [i for i in range(index - offset, index + offset) if i >= 0 and i < len(matrix)]
        for ind in indexes:
            segmented = initial_segmentation(matrix[ind], 40)
            denoised = denoise(segmented)
            neigh.append(denoised)
        
        neigh = np.array(neigh)
        return np.logical_or.reduce(neigh)*255
    
    segmented = initial_segmentation(matrix[index], 40)
    return denoise(segmented)

def segmentation(index, matrix, save=False):
    if save: save_image(matrix[index], f'{PATH}normal_{index}.png')
    mask = get_mask(index, matrix, OR=False)
    if save: save_image(mask, f'{PATH}mask_{index}.png')
    regions = get_regions(mask)
    return regions

def get_regions(img):
    label_image = label(img)
    regions = regionprops(label_image)
    return label_image, regions

def latest_region(regions):
    max_region = 0
    for i, region in regions.keys():
        if region > max_region:
            max_region = region
    return max_region

def segmentate_matrix(matrix):
    found_regions = {}
    # for i in range(len(matrix)):
    for i in range(5):
        labels, regions = segmentation(i, matrix)
        max_region = latest_region(found_regions)
        for region in regions:
            labels[labels == region.label] = region.label + max_region
            region.label = region.label + max_region
            found_regions[(i, region.label)] = [labels, region]
    print(found_regions)
