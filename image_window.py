import os
from itertools import product

import cv2
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt
import numpy as np

from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit

import argparse
import random as rng

from PIL import Image
from skimage.filters import rank
from skimage.morphology import disk
from tqdm import tqdm


def random_colors(n_color, add_a=False):
    color_list = []
    for idx in range(n_color):
        c = list(np.random.random_integers(0, 255, size=3))
        if add_a:
            c.append(255)
        color_list.append(tuple(np.array(c, dtype=np.uint8)))
    return color_list


if __name__ == "__main__":
    fileDir = os.path.dirname(os.path.realpath('_file_'))
    print(fileDir)
    filename = "005center-1.TIFF"
    dir_name = ''
    file_dir = os.path.join(os.getcwd() + dir_name, filename)
    print(file_dir)
    data = Image.open(file_dir)
    data = (np.array(data) / 256).astype('uint8')
    print(data.shape)
    w, h = data.shape

    # Get lines
    line_th = 70
    idxs_x, idxs_y = np.where(data > line_th)
    line_filtered = np.zeros(data.shape, dtype=np.uint8)
    for x, y in zip(idxs_x, idxs_y):
        line_filtered[x, y] = data[x, y]

    # Obtain junctions by local averaged threshold
    junction_mask = np.zeros(data.shape, dtype=np.uint8)
    win_size = 30
    it_over = product(range(0, w - win_size, win_size), range(0, h - win_size, win_size))
    print((w - win_size) * (h - win_size))
    for i, j in tqdm(it_over):
        x1, y1, x2, y2 = i, j, min(i + win_size, w), min(j + win_size, h)
        wind_data = line_filtered[x1:x2, y1:y2]
        n_px = len(np.where(wind_data > line_th)[0])
        if n_px == 0:
            continue
        avg = np.sum(wind_data) / n_px
        th = avg * 1.0
        for x, y in product(range(x1, x2), range(y1, y2)):
            if data[x, y] > th:
                junction_mask[x, y] = 255
    '''
    radius = 30
    selem = disk(radius)
    local_otsu = rank.otsu(data, selem)
    
    data_binary = (data >= (local_otsu * 4)) * 255
    '''

    # Aggregate nearest points
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(junction_mask, connectivity=8)
    print("Number of junctions: ", retval - 1)

    # Draw separated junction points
    color_list = random_colors(retval, add_a=True)
    junction_point_image = np.zeros((w, h, 4), dtype=np.uint8)
    for x, y in product(range(w), range(h)):
        idx_obj = labels[x, y]
        if idx_obj > 0:
            color = color_list[idx_obj]
            junction_point_image[x, y] = np.array(color, dtype=np.uint8)

    # Draw all
    color_data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    plt.imshow(color_data)
    # plt.imshow(junction_point_image)
    x_list, y_list = list(), list()
    for x, y in centroids[1:]:
        x_list.append(x)
        y_list.append(y)
    point_size = 3
    plt.scatter(x_list, y_list, s=10, c=[(r / 256, g / 256, b / 256, a / 256) for r, g, b, a in color_list[1:]])
    plt.show()

    # Interactive
    max_items = 100000
    color_list = color_list + random_colors(max_items)
    updated_centroids = {i + 1: (int(x), int(y)) for i, (x, y) in enumerate(centroids[1:])}
    current_centroids_idx = list(updated_centroids.keys())
    original_max_idx = len(updated_centroids)

    idx_file = None
    def draw_centroids(save=False, overwrite=True, idx_file=None):
        new_img = color_data.copy()
        for idx in current_centroids_idx:
            x_c, y_c = updated_centroids[idx]
            c = tuple([int(u) for u in color_list[idx]])
            cv2.circle(new_img, (x_c, y_c), radius=point_size, color=c)
        cv2.imshow('Interactive', new_img)
        print(f"updated junctions : {len(current_centroids_idx)}")
        if idx_file is None:
            idx_file = 0
        if save:
            ex = file_dir.split('.')[-1]
            new_filename = '.'.join(file_dir.split('.')[:-1])+f'_updated_{idx_file}.'+ex
            while os.path.isfile(new_filename) and not overwrite:
                idx_file += 1
                new_filename = '.'.join(file_dir.split('.')[:-1])+f'_updated_{idx_file}.'+ex
            cv2.imwrite(new_filename, new_img)
        return idx_file

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, (x_c, y_c) in updated_centroids.items():
                if (x - x_c) ** 2 + (y - y_c) ** 2 < point_size ** 2:
                    if idx not in current_centroids_idx:
                        current_centroids_idx.append(idx)
                        break
            else:
                new_idx = max(updated_centroids.keys()) + 1
                updated_centroids[new_idx] = x, y
                current_centroids_idx.append(new_idx)
            draw_centroids(save=True, overwrite=True, idx_file=idx_file)
        elif event == cv2.EVENT_RBUTTONDOWN:
            for idx, (x_c, y_c) in updated_centroids.items():
                if (x - x_c) ** 2 + (y - y_c) ** 2 < point_size ** 2:
                    if idx in current_centroids_idx:
                        current_centroids_idx.remove(idx)
                        if idx > original_max_idx:
                            updated_centroids.pop(idx)
                        break
            draw_centroids(save=True, overwrite=True, idx_file=idx_file)
        elif event == cv2.EVENT_MBUTTONDOWN:
            draw_centroids(save=True, overwrite=True, idx_file=idx_file)

    idx_file = draw_centroids(save=True, overwrite=False, idx_file=None)
    print(idx_file)
    cv2.setMouseCallback('Interactive', click_event)
    cv2.waitKey(0)
