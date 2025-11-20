import os 
import matplotlib.pyplot as plt

import cv2
import numpy as np


def load_img(img_path, greyscale=True): 
    """ This function serves as a dataloader, loads img from img_path as grayscale """
    flag = cv2.IMREAD_ANYCOLOR
    if greyscale:
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


