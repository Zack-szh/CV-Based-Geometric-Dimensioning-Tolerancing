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

def load_img_from_folder(folder_path): 
    """Loads images from folder into a list """
    