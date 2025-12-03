import main
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import math
from skimage.measure import ransac, CircleModel
from sklearn.cluster import DBSCAN

from enum import Enum, auto

class DETECTION(Enum):
    CIRCLE = auto()
    LINE = auto()



# ----------------------------------------------------------------------------------------------------
# ---------- DETECTION METHODS
# ----------------------------------------------------------------------------------------------------

def detect_line(img, canny_thres1=50, 
             canny_thres2=150, 
             rho=1, 
             theta=np.pi/180, 
             hough_thres=80, 
             min_line_len=30, 
             max_line_gap=10
             ): 
    """The line detection pipeline"""

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

def applyHoughTransform(image):
    if image.ndim == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        base_for_drawing = image.copy()
    else:
        img_gray = image
        base_for_drawing = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lines = detect_line(
        img_gray,
        canny_thres1=100,
        canny_thres2=200,
        rho=1,
        theta=main.np.pi/180,
        hough_thres=200,
        min_line_len=10,
        max_line_gap=1e6,
    )

    img_with_lines = draw_lines(base_for_drawing, lines, color=(0, 0, 255), thickness=2)

    img_rgb = cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB)

    return img_rgb

def find_lines(input, canny=False, probabilistic=True):
    img = input.copy()

    if canny:
        img = cv2.Canny(img, 50, 150)

    rho = 1
    theta = np.pi / 180
    threshold = 200

    if probabilistic:
        raw_lines = cv2.HoughLinesP(
            img,
            rho=rho,
            theta=theta,
            threshold=threshold,
            minLineLength=10,
            maxLineGap=1_000_000
        )

        if raw_lines is None:
            print("found 0 lines")
            lines = []
        else:
            lines = [tuple(line[0]) for line in raw_lines]

    else:
        lns = cv2.HoughLines(img, rho=rho, theta=theta, threshold=threshold,
                             srn=0, stn=0)
        lines = []
        if lns is not None:
            for i in range(len(lns)):
                rho_i = lns[i][0][0]
                theta_i = lns[i][0][1]
                a = math.cos(theta_i)
                b = math.sin(theta_i)
                x0 = a * rho_i
                y0 = b * rho_i
                lines.append(
                    (int(x0 + 1000 * (-b)), int(y0 + 1000 * a),
                     int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
                )

        print(f"found {len(lines)} lines")

    base = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
    input_n_lines = draw_lines(base, lines)
    return cv2.cvtColor(input_n_lines, cv2.COLOR_BGR2RGB)


def find_lines_HT(edges, return_img=False):
    """ find lines from edge map using Hough Transform; 
        consider integrating into detection.py
    """
    
    # Detect lines using HT
    lines = cv2.HoughLinesP(edges,
                        1, np.pi/180,
                        threshold=500,    # threshold effectiveness influenced by thickness of lines, but not sensitive
                        minLineLength=10,
                        maxLineGap=1e6)

    # Check for no lines found
    if lines is None:
        return []
    else:
        return [tuple(line[0]) for line in lines]


def find_circles(input,
                 blur=True,
                 thresh=60, #was 40 for raw image
                 dp=1.2,
                 minDist=20,
                 minRadius=20,
                 maxRadius=0,
                 param1=100,
                 param2=None,
                 return_type: str = "circles"):
    """Detect circles using cv2.HoughCircles.

    Parameters mirror the commonly tuned Hough params. By default, returns an
    RGB image with circles drawn (backward compatible). Set return_type="circles"
    to return a list of (x, y, r) integer tuples suitable for downstream matching.
    """
    img = input.copy()

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if blur:
        img = cv2.medianBlur(img, 17)

    p2 = thresh if param2 is None else param2

    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=p2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    if return_type == "circles":
        if circles is None:
            return []
        circles = np.round(circles[0, :]).astype(int)
        return [tuple(c) for c in circles]

    # default: return an RGB image with circles drawn (legacy behavior)
    base = input.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for (x, y, r) in circles:
            cv2.circle(base, (x, y), r, (0, 255, 0), 2)
            cv2.circle(base, (x, y), 2, (0, 0, 255), 3)

    return base


def find_circles_HT(edges, R_expect=(100,133,167)):   # use (100, 33, 133, 167) to find servo mount points
    """ find circles from edge map using Hough Transform; 
        radius to look for can be specified by R_expect, which can be either int or tuple of int
    """
    # Check if there is an expected radius
    if R_expect is not None:
        if type(R_expect) == int:     # If a single radius is specified
            # print(R_expect)
            r_min = R_expect - 10
            r_min = max(r_min, 1)
            r_max = R_expect + 10
        elif type(R_expect) == tuple:     # If multiple radii are specified
            circles = []
            for r in R_expect:
                circles.extend(find_circles_HT(edges, R_expect=r))
            return circles
    else:
        r_min = 1
        r_max = 1000

    # Detect circles using HT
    # circles = cv2.HoughCircles(edges,
    #                         method=cv2.HOUGH_GRADIENT,
    #                         dp=1,
    #                         param1=100,
    #                         param2=20,     # param2 effectiveness influenced by completeness & thickness of circles; pretty sensitve, and seem especially sensitive to salt & pepper noise
    #                         minDist=2,
    #                         minRadius=r_min,
    #                         maxRadius=r_max)
    
    # Detect circles using HT
    circles = cv2.HoughCircles(edges,
                            method=cv2.HOUGH_GRADIENT_ALT,
                            dp=1,
                            param1=100,
                            param2=0.3, # was 0.3
                            minDist=2,
                            minRadius=r_min,
                            maxRadius=r_max)

    # Check for no circles found
    if circles is None:
        return []
    else:
        return [(int(c[0]), int(c[1]), int(c[2])) for c in circles[0]]


def find_circles_contours(input, filter=30):
    img = input.copy()

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < filter:
            continue

        (x, y), r = cv2.minEnclosingCircle(c)
        circle_area = np.pi * r * r
        circularity = area / circle_area

        if 0 < circularity < 3:
            circles.append((int(x), int(y), int(r)))

    print(f"found {len(circles)} circles")

    base = input.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

    for (x, y, r) in circles:
        cv2.circle(base, (x, y), r, (0, 255, 0), 2)
        cv2.circle(base, (x, y), 2, (0, 0, 255), 9)

    return base


def find_circle_ransac(input_gray):
    if len(input_gray.shape) == 3:
        input_gray = cv2.cvtColor(input_gray, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(input_gray, (11, 11), 0)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 30, 100)    
    ys, xs = np.nonzero(edges)
    points = np.column_stack([xs, ys])

    if len(points) < 3:
        print("Not enough edge points")
        return input_gray

    model, inliers = ransac(
        points,
        CircleModel,
        min_samples=3,
        residual_threshold=2.0,
        max_trials=1000
    )

    xc, yc, r = model.params

    # 3. Draw circle
    base = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.circle(base, (int(xc), int(yc)), int(r), (0,255,0), 2)

    return cv2.cvtColor(base, cv2.COLOR_BGR2RGB)

# ----------------------------------------------------------------------------------------------------
# ---------- FILTRATION METHODS
# ----------------------------------------------------------------------------------------------------

def filter_lines(lines, eps_pos=10.0, eps_len=20.0, eps_angle=0.1, min_samples=2):
    if not lines:
        return []

    cpfs = []
    for (x1, y1, x2, y2) in lines:
        x0 = (x1 + x2) / 2.0
        y0 = (y1 + y2) / 2.0
        L  = math.hypot(x2 - x1, y2 - y1)
        t  = math.atan2(y2 - y1, x2 - x1)
        if t < 0:
            t += math.pi
        cpfs.append((x0, y0, L, t))

    cpfs = np.array(cpfs, dtype=float)

    # Scale features for DBSCAN
    X = np.empty_like(cpfs)
    X[:, 0] = cpfs[:, 0] / eps_pos
    X[:, 1] = cpfs[:, 1] / eps_pos
    X[:, 2] = cpfs[:, 2] / eps_len
    X[:, 3] = cpfs[:, 3] / eps_angle

    db = DBSCAN(eps=3.0, min_samples=min_samples)
    labels = db.fit_predict(X)

    filtered = []
    unique_labels = set(labels)
    for lbl in unique_labels:
        if lbl == -1:
            continue

        cluster_cpfs = cpfs[labels == lbl]
        if cluster_cpfs.size == 0:
            continue

        x0 = np.median(cluster_cpfs[:, 0])
        y0 = np.median(cluster_cpfs[:, 1])
        L  = np.max(cluster_cpfs[:, 2])
        t  = np.median(cluster_cpfs[:, 3])

        # Convert back to (x1, y1, x2, y2)
        dx = (L / 2.0) * math.cos(t)
        dy = (L / 2.0) * math.sin(t)

        x1 = int(round(x0 - dx))
        y1 = int(round(y0 - dy))
        x2 = int(round(x0 + dx))
        y2 = int(round(y0 + dy))

        filtered.append((x1, y1, x2, y2))

    return filtered


def filter_circles(edges, circles, height, width,
                   radial_tolerance=1, # Assumed thickness tolerance of an edge 
                   grad_mag_thresh=6.0, # Magnitude threshold for gradient difference
                   angle_cos_thresh=0.7, # Threshold from orthogonal to test 
                   min_strength=0.2, # Minimum required strength to allow a cluster (0-1)
                   num_samples=180): # Num point samples drawn along the cricle

    canvas = np.zeros((height, width), dtype=np.uint8)

    # Normalize circles to a Python list of tuples
    if circles is None:
        circles = []
    elif isinstance(circles, np.ndarray):
        circles = circles.tolist()

    if hasattr(circles, '__len__') and len(circles) == 0:
        return [], canvas

    # Edge Gradients
    gx = cv2.Sobel(edges, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(edges, cv2.CV_32F, 0, 1, ksize=3)

    circles = [(int(x), int(y), int(r)) for (x, y, r) in circles if r > 0]
    n = len(circles)
    if n == 0:
        return [], canvas

    # Group circles by containment w/ union find (borrowed implementation)

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    def contains(ci, cj):
        xi, yi, ri = ci
        xj, yj, rj = cj
        dx = xj - xi
        dy = yj - yi
        d = math.hypot(dx, dy)
        # return d + rj <= ri
        return d <= ri

    for i in range(n):
        for j in range(i + 1, n):
            ci = circles[i]
            cj = circles[j]
            if contains(ci, cj) or contains(cj, ci):
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Strength scoring using edge gradients
    def circle_strength(circle):
        x, y, r = circle
        if r <= 0:
            return 0.0

        thetas = np.linspace(0, 2 * np.pi, num_samples, endpoint=False) # Sampling points
        good = 0 # Num points that show gradient along drawn circle that is orthogonal to diameter
        total = len(thetas)

        h, w = edges.shape[:2]

        for theta in thetas:
            ct = math.cos(theta)
            st = math.sin(theta)

            for dr in range(-radial_tolerance, radial_tolerance + 1):
                rr = r + dr # Adding tolerance to assume some thickness in edge map / gradient
                if rr <= 0:
                    continue
                # Get point x and point y given x,y radius and angle of approach 
                px = int(round(x + rr * ct)) 
                py = int(round(y + rr * st))

                if px < 0 or px >= w or py < 0 or py >= h:
                    continue

                gx_val = float(gx[py, px])
                gy_val = float(gy[py, px])
                mag = math.hypot(gx_val, gy_val)

                if mag < grad_mag_thresh:
                    continue

                # Normalize and get sobel vector (with some additional numbers to prevent divide by zero)
                ux, uy = gx_val / (mag + 1e-6), gy_val / (mag + 1e-6)

                # Get angle of vector w/ relation to circle with dot product
                cos_angle = abs(ux * ct + uy * st)

                if cos_angle >= angle_cos_thresh:
                    good += 1
                    break

        return good / total

    kept_circles = []

    for root, idxs in groups.items():
        # Evaluate strength for each circle in this group
        best_circle = None
        best_score = -1.0

        for idx in idxs:
            c = circles[idx]
            score = circle_strength(c)
            if score > best_score:
                best_score = score
                best_circle = c

        # Keep strongest circle if it meets threshold
        if best_circle is not None and best_score >= min_strength:
            kept_circles.append(best_circle)

    # Draw kept circles on canvas for debugging / visualization
    for x, y, r in kept_circles:
        cv2.circle(canvas, (x, y), r, 255, thickness=1)

    return kept_circles, canvas



# ----------------------------------------------------------------------------------------------------
# ---------- DISPLAY METHODS
# ----------------------------------------------------------------------------------------------------

def display_single_image(image, title):
    main.plt.title(title)
    main.plt.imshow(image)
    main.plt.axis('off')
    main.plt.show()
    

def sample_files(folder_path, n):
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    if n > len(files):
        raise ValueError(f"Requested {n} files, but only {len(files)} available.")

    return random.sample(files, n)
    
def show_images_from_files(files, title, image_proc=None, cols=4, figsize=(12, 10)):
    rows = math.ceil(len(files) / cols)
    plt.figure(figsize=figsize)

    imgs = []
    for i, file in enumerate(files):
        img = cv2.imread(file)
        imgs.append(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if image_proc is not None:
            img = image_proc(img)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(title[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    return imgs

def subsample_technique(technique, title, image_proc = lambda img: img, generated_num = 3, realistic_num=1):
    base_path = os.getcwd()
    files = []
    img_path_physical = os.path.join(base_path, "data", "physical parts")
    img_path = os.path.join(base_path, "data", "geometric shapes dataset")

    match (technique):
        case DETECTION.CIRCLE:
            img_path_circle = os.path.join(img_path, "Circle")
            files += sample_files(img_path_physical, realistic_num) + sample_files(img_path_circle, generated_num)
        case DETECTION.LINE:
            img_path_square = os.path.join(img_path, "Square")
            img_path_triangle = os.path.join(img_path, "Triangle")
            files += sample_files(img_path_physical, realistic_num) + sample_files(img_path_square, generated_num//2) + sample_files(img_path_triangle, generated_num//2)
        case _:
            print("Default case reached")

    show_images_from_files(files, title, image_proc)

    




