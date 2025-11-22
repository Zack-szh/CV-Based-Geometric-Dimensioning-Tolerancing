import numpy as np
import cv2



# ----------------------------------------------------------------------------------------------------
# ---------- KEY FUNCTIONS
# ----------------------------------------------------------------------------------------------------

def gradient(img, derv_len=1, use_sobel=False):
    """ takes gradient of the input image (img); returns superscored edge map and directional derivatives.
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

def applyGradient(image):
    sigma = 30
    thrs = 97
    img_part1_gb = cv2.GaussianBlur(image, ksize=(sigma+1, sigma+1), sigmaX=sigma, sigmaY=sigma)
    edges_crude, Ix, Iy = gradient(img_part1_gb, derv_len=sigma//10)
    edges_crude = np.where(edges_crude >= np.percentile(edges_crude, thrs), 255, 0).astype("uint8")
    return edges_crude

def getEdgesMasked(img):
    # get fine edges (org image, 1-step gradient)
    edges_fine, Ix, Iy = pp.gradient(img, derv_len=1)
    # show_edge_results(img, edges_fine, Ix, Iy)

    # gauss blur img
    sigma = 50
    img_part1_gb = cv2.GaussianBlur(img, ksize=(sigma+1,sigma+1), sigmaX=sigma, sigmaY=sigma)
    # take large-step gradient
    step = sigma//10
    step = step if step>0 else 1
    edges_crude, Ix, Iy = pp.gradient(img_part1_gb, derv_len=sigma//10)
    # threshold
    thrs = 95   # percentile above which to keep
    edges_crude = np.where(edges_crude>=np.percentile(edges_crude, thrs), 1, 0)
    # show_edge_results(img_part1_gb, edges_crude, Ix, Iy)
    # mask
    edges_masked = np.where(edges_crude>0, edges_fine, 0)

    return edges_masked, Ix, Iy