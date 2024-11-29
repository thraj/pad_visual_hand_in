
import sys
sys.path.append('/home/maria/Documents/projects/avs7_miniproject/')

import numpy as np
from PIL import Image
import random
import scipy as sp

from constants import MVTEC, GAMMA_PARAMS, OBJECT_RESIZE_BOUNDS, TEXTURE_RESIZE_BOUNDS
from utils.visualization import *


def read_image(path_to_image: str, mask_image: bool, scale: bool=False):
    
    img = Image.open(path_to_image)

    if mask_image:
        img = img.convert("L") # greyscale
        binary_mask = np.array(img) > 127
        return binary_mask.astype(np.uint8)

    img = np.array(img.convert('RGB'))
    if scale:
        return img.astype('double') / 255.0

    return img

def get_neighbors(i: int, j: int, max_i: int, max_j: int):
    return [(i + di, j) for di in (-1, 1) if 0 <= i + di <= max_i] + \
        [(i, j + dj) for dj in (-1, 1) if 0 <= j + dj <= max_j]
        

def populate_normal(A, b, y_coords, x_coords, pixel_idx_map, src_image_test, dest_image_test, H, W):

    counter = 0
    num_mask_pixels = len(y_coords)

    for index in range(num_mask_pixels):
        y, x = y_coords[index], x_coords[index]

        for ny, nx in get_neighbors(y, x, H-1, W-1):
            A[counter, pixel_idx_map[y][x]] = 1
            
            b[counter] = src_image_test[y][x] - src_image_test[ny][nx]

            if pixel_idx_map[ny][nx] != -1:
                A[counter, pixel_idx_map[ny][nx]] = -1
            else:
                b[counter] += dest_image_test[ny][nx]

            counter += 1
    
    return A, b

def populate_mixed(A, b, y_coords, x_coords, pixel_idx_map, src_image_test, dest_image_test, H, W):

    counter = 0
    num_mask_pixels = len(y_coords)

    for index in range(num_mask_pixels):
        y, x = y_coords[index], x_coords[index]

        for ny, nx in get_neighbors(y, x, H-1, W-1):
            d1 = src_image_test[y][x] - src_image_test[ny][nx]
            d2 = dest_image_test[y][x] - dest_image_test[ny][nx]

            strongest = d1 if abs(d1) > abs(d2) else d2

            A[counter, pixel_idx_map[y][x]] = 1
            
            b[counter] = strongest

            if pixel_idx_map[ny][nx] != -1:
                A[counter, pixel_idx_map[ny][nx]] = -1
            else:
                b[counter] += dest_image_test[ny][nx]

            counter += 1
    
    return A, b

def compute_poisson_blend_channel(src_image_test, dest_image_test, mask, mode):

    H, W = src_image_test.shape

    num_mask_pixels = mask.sum().astype(int)
    pixel_idx_map = np.full(mask.shape, -1, dtype=int)
    y_coords, x_coords = np.where(mask == 1)
    pixel_idx_map[mask > 0] = np.arange(num_mask_pixels)

    A = sp.sparse.lil_matrix((4 * num_mask_pixels, num_mask_pixels))
    b = np.zeros(4 * num_mask_pixels)

    if mode=='normal':
        A, b = populate_normal(A, b, y_coords, x_coords, pixel_idx_map, src_image_test, dest_image_test, H, W)

    if mode=='mixed':
        A, b = populate_mixed(A, b, y_coords, x_coords, pixel_idx_map, src_image_test, dest_image_test, H, W)


    A = sp.sparse.csr_matrix(A)
    v = sp.sparse.linalg.lsqr(A, b)[0]

    copy_dest = dest_image_test.copy()

    for index in range(num_mask_pixels):
        y, x = y_coords[index], x_coords[index]
        copy_dest[y][x] = v[pixel_idx_map[y][x]]

    
    return np.clip(copy_dest, 0, 1)