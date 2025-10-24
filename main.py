import os 
import matplotlib.pyplot as plt

import cv2
import numpy as np


def load_img(img_path): 
    """This function serves as a dataloader, loads img from img_path as grayscale"""
    flag = cv2.IMREAD_GRAYSCALE
    img = cv2.imread(img_path, flag)    # loads img as grayscale, one channel to work with
    
    if img is None: 
        raise FileNotFoundError(f"File not found at: {img_path}")
    
    return img


def load_img_from_folder(folder_path, batch_size = 10): 
    """Loads images from folder into a list """

    supported_format = ('jpg', 'jpeg', 'png')
    data = []
    count = 0

    for f in os.listdir(folder_path): 
        if f.endswith(supported_format) and count < batch_size: 
            img_path = os.path.join(folder_path, f)
            img = load_img(img_path)
            data.append(img)
            count += 1
        else: 
            break

    return data


def detect_line(img, canny_thres1=50, 
             canny_thres2=150, 
             rho=1, 
             theta=np.pi/180, 
             hough_thres=80, 
             min_line_len=30, 
             max_line_gap=10
             ): 
    """The line detection pipeline"""

    # the images are already loaded in as grayscale

    # step 1: canny edge detections
    edges = cv2.Canny(img, canny_thres1, canny_thres2)

    # step 2: detect lines
    lines = cv2.HoughLinesP(edges, rho, theta, hough_thres, minLineLength=min_line_len, maxLineGap=max_line_gap)

    if lines is None:
        return []

    return [tuple(line[0]) for line in lines]  # flatten


def draw_lines(img, lines, color=(0, 0, 255), thickness=2): 
    img_out = img.copy()
    for (x1, y1, x2, y2) in lines: 
        cv2.line(img_out, (x1, y1), (x2, y2), color, thickness)

    return img_out