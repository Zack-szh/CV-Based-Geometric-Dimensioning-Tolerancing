import numpy as np
import cv2
import main
from scipy import fft


# ----------------------------------------------------------------------------------------------------
# ---------- KEY FUNCTIONS
# ----------------------------------------------------------------------------------------------------

def get_edges(original):
    """ converts original (raw) GRAYSCALE image into edge map
    """

    """ Try this function with these params for Hough circle detection:
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        param1=100,
        param2=40,
        minDist=2*10,
        minRadius=10,
        maxRadius=0
    """

    # 1. LPF: gauss blur
    k = 15
    gauss = cv2.GaussianBlur(original, ksize=(2*k+1,2*k+1), sigmaX=2*k, sigmaY=2*k)
    # 2. edge finding: gradient
    grad,_,_ = gradient(gauss)
    # 3. refinement
    # threshold
    p = 90
    thrs = np.where(grad > np.percentile(grad, p), 255, 0).astype(np.uint8)
    # erode
    k = 4
    eroded = cv2.erode(thrs, np.ones((k, k)))

    return eroded


def gradient(img, derv_len=1, use_sobel=False):
    """ takes gradient of the input image (img) using "step masks" of step length derv_len; 
        superscores gradient accross each color channel and each direction to get final gradient;
        img can have 1 channel (greyscale) or 3 channels (color)
    """
    # check input dimensions
    shape = img.shape
    combine_channels = True     # flag: perform edge combination along channels
    if len(shape) == 2:     # grey scale
        combine_channels = False
        
    # define derivative filters
    if use_sobel:
        dfdx = cv2.Sobel(img, ddepth=-1, dx=1, dy=0, ksize=2*derv_len+1)
        dfdy = cv2.Sobel(img, ddepth=-1, dx=0, dy=1, ksize=2*derv_len+1)
    else:  # use simple derivative
        # dx = np.array([[-1, 1]])
        # dy = np.array([[1, -1]]).T
        ones = np.repeat(1, derv_len).reshape(1, derv_len)
        dx = np.concatenate([-ones, ones], axis=1)
        dy = np.concatenate([ones, -ones], axis=1).T

        # differentiate each color channel on each axis on both directions
        # on both directions (pos & neg) b/c we are dealing w/ uint8
        dfdx_pos = cv2.filter2D(src=img, ddepth=-1, kernel=dx)
        dfdy_pos = cv2.filter2D(src=img, ddepth=-1, kernel=dy)
        dfdx_neg = cv2.filter2D(src=img, ddepth=-1, kernel=np.flip(dx))
        dfdy_neg = cv2.filter2D(src=img, ddepth=-1, kernel=np.flip(dy))
    
        # merge pos & neg results on each axis
        dfdx = np.maximum(dfdx_pos, dfdx_neg)
        dfdy = np.maximum(dfdy_pos, dfdy_neg)

    # combine edge results on each direction
    out = np.maximum(dfdx, dfdy)
    # out = np.sqrt( np.power(dfdx, 2) + np.power(dfdy, 2) )

    # combine edge results on each channel
    if combine_channels:
        out = np.max(out, axis=2)

    return out, dfdx, dfdy


def fft_LPF(img, r_cutoff=0.95, verbose=False):
    """ performs low pass filtering by: 1. taking fft on img (size: H,W), 2. applying a rectangular binary mask in freq domain, 
        and 3. inverse fft the masked result back into position domain to get filtered image.
        r_cutoff (float in [0,1]) defines the size of the rectangular freq domain mask 
        (height = H*r_cutoff, width = W*r_cutoff, centered at origin of freq domain);
        only components inside the mask are kept  
    """

    # get image size
    if img.ndim == 3:   # colored image (BGR)
        H,W,C = img.shape
    else:   # single channel (grey scale)
        H,W = img.shape
        C = 1

    # find center coord
    hc = (H-1)/2.
    wc = (W-1)/2.

    # 2D fourier transform
    f_img = fft.fft2(img)

    # shift 0-freq to center of f_img
    f_img = fft.fftshift(f_img)

    # define frequency domain mask  
    # find cutoff boundaries
    wl = int(wc * r_cutoff)
    wu = int(wc * (2 - r_cutoff)) +1
    hl = int(hc * r_cutoff)
    hu = int(hc * (2 - r_cutoff)) +1
    # build mask
    mask = np.zeros_like(img)
    mask[hl:hu+1 , wl:wu+1] = 1

    # # TEST: guass blur mask
    # k=100
    # mask = cv2.GaussianBlur(mask, ksize=(2*k+1,2*k+1), sigmaX=k, sigmaY=k)

    if verbose:
        print(f"input image size: W,H = ({W}, {H})")
        print(f"cutoff limits: width:({wl}, {wu}), height:({hl}, {hu})")

    # apply mask
    f_filtered = f_img * mask
    # f_filtered = f_img

    if verbose:
        # TODO: show freq domain filtered results
        pass

    # inverse 2D fourier transform
    filtered = fft.ifft2(f_filtered)
    # filtered = fft.fftshift(filtered)
    filtered = np.abs(filtered).astype(np.uint8)

    return filtered


# ----------------------------------------------------------------------------------------------------
# ---------- ARCHIVED (rough coded functions for quick testing; kept to not break things)
# ----------------------------------------------------------------------------------------------------

def crop(img, Z=4):
    # crop img
    H,W = img.shape[0:2]
    Zh = Z//2   # Z is the zoom factor
    img = img[H*(Zh-1)//Z : H*(Zh+1)//Z , W*(Zh-1)//Z : W*(Zh+1)//Z]
    # convert to RGB
    img = main.cv2.cvtColor(img, main.cv2.COLOR_BGR2RGB)
    return img

def crop_largest_structure(img, min_area=500, margin=10):
    image = img.copy()

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray_blur, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        h, w = image.shape[:2]
        return image, (0, 0, w, h)

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        h, w = image.shape[:2]
        return image, (0, 0, w, h)

    x, y, w, h = cv2.boundingRect(largest)

    x = max(x - margin, 0)
    y = max(y - margin, 0)
    x2 = min(x + w + 2 * margin, image.shape[1])
    y2 = min(y + h + 2 * margin, image.shape[0])

    cropped = image[y:y2, x:x2]
    return cropped, (x, y, x2 - x, y2 - y)

def applyGradient(image):
    sigma = 30
    thrs = 97
    img_part1_gb = cv2.GaussianBlur(image, ksize=(sigma+1, sigma+1), sigmaX=sigma, sigmaY=sigma)
    edges_crude, Ix, Iy = gradient(img_part1_gb, derv_len=sigma//10)
    edges_crude = np.where(edges_crude >= np.percentile(edges_crude, thrs), 255, 0).astype("uint8")
    return edges_crude

def getEdgesMasked(img):
    # get fine edges (org image, 1-step gradient)
    edges_fine, Ix, Iy = gradient(img, derv_len=1)
    # show_edge_results(img, edges_fine, Ix, Iy)

    # gauss blur img
    sigma = 50
    img_part1_gb = cv2.GaussianBlur(img, ksize=(sigma+1,sigma+1), sigmaX=sigma, sigmaY=sigma)
    # take large-step gradient
    step = sigma//10
    step = step if step>0 else 1
    edges_crude, Ix, Iy = gradient(img_part1_gb, derv_len=sigma//10)
    # threshold
    thrs = 95   # percentile above which to keep
    edges_crude = np.where(edges_crude>=np.percentile(edges_crude, thrs), 1, 0)
    # show_edge_results(img_part1_gb, edges_crude, Ix, Iy)
    # mask
    edges_masked = np.where(edges_crude>0, edges_fine, 0)

    return edges_masked, Ix, Iy