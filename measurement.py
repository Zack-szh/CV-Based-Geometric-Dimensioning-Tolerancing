import main
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import math
from skimage.measure import ransac, CircleModel


def line_measurement(line, conversion_ratio=1.0) -> list: 
    """
    Given a list of lines in the format [(x1, y1, x2, y2), ...], 
    we compute length of each line using the given conversion ratio
    """

    measured_lines = [] 

    for x1, y1, x2, y2 in line: 
        pixel_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        real_length = pixel_length * conversion_ratio
        measured_lines.append(real_length)

    return measured_lines

def circle_measurement(circle, conversion_ratio=1.0)-> list: 
    """
    Given a list of circles in the format 
    """
    pass

