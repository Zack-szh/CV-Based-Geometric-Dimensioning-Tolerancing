import main
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import math
from skimage.measure import ransac, CircleModel


def measure_line(line, conversion_ratio=1.0): 
    """
    Given a list of lines in the format [(x1, y1, x2, y2), ...], 
    we compute length of each line using the given conversion ratio, 
    output: [(x1, y1, x2, y2, length), ...]
    """

    measured_lines = [] 

    for x1, y1, x2, y2 in line: 
        pixel_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        real_length = pixel_length * conversion_ratio
        measured_lines.append((x1, y1, x2, y2, real_length))

    return measured_lines


def measure_circle(circle, conversion_ratio=1.0): 
    """
    Given a list of circles in the format [(x0, y0, r), ...], 
    we compute radius of each circle using the given conversion ratio
    output: [(x0, y0, real_radius), ...]
    """
    
    measured_circles = []

    for x0, y0, r in circle: 
        real_radius = r * conversion_ratio
        measured_circles.append((x0, y0, real_radius))

    return measured_circles


def draw_measurements(img, lines, circles): 
    """
    Given measured_lines and measured_circles, draw them on img, 
    returns img with measurements drawn
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    color = (255, 255, 0)  
    thickness = 5
    output = img.copy()
    unit = "mm"

    for x1, y1, x2, y2, length in lines: 
        # for lines, we draw the length at the midpoint 
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        text = f"{length:.2f} {unit}"

        cv2.line(output, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(output, text, (mid_x + 10, mid_y + 10), font, font_scale, color, thickness, cv2.LINE_AA)

    for x0, y0, radius in circles: 
        # for circles, we draw the radius at the center
        text = f"r={radius:.2f} {unit}"

        cv2.circle(output, (x0, y0), int(radius), color, 3)
        
        cv2.circle(output, (x0, y0), 5, (0, 0, 255), -1)

        cv2.putText(output, text, (x0 + 15, y0 - 15), font, font_scale, color, thickness, cv2.LINE_AA)
        
        
    return output
