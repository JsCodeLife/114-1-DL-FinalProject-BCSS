import numpy as np
from PIL import Image
import os
import cv2

def load_image_and_mask(img_path, mask_path):
    """
    Load image and mask.
    Image: RGB
    Mask: grayscale (preserved values)
    """
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path) # Keep original mode/values
    return np.array(img), np.array(mask)

def apply_background_whitening(img, mask):
    """
    Set pixels where mask == 0 to pure white (255, 255, 255).
    """
    img = img.copy()
    # Mask is 2D (H, W), Image is 3D (H, W, 3)
    # Background (label 0) -> White
    img[mask == 0] = [255, 255, 255]
    return img

def compute_stats(image_paths_list):
    """
    Compute global mean and std of R,G,B channels from a list of image paths.
    This assumes images are already processed (and potentially whitened).
    If we want to ignore background (white pixels) in stats, we should check for [255,255,255].
    HOWEVER, for standard normalization (transforms.Normalize), we usually normalize the WHOLE image 
    submitted to the network. If the network sees white background, that background should be part of the stats 
    OR we accept that (255-mean)/std is a specific value.
    
    Standard practice: Compute stats on the dataset as it enters the network. 
    If we feed white background, we include it. 
    BUT, often we mask it out for Staining Normalization (Macenko).
    For Z-score input normalization, PyTorch usually expects stats of the valid content distributions 
    or the whole image distribution. 
    Given we force background to 255, the distribution will have a huge spike at 255.
    
    Let's compute stats of the ROI ONLY (non-background) to be safe?
    User said: "把這兩個數字寫在一個 txt 或 json 給組員... 加一行 transforms.Normalize".
    If they use Normalize, it shifts everything. If we include 255s in mean calculation, mean will be high.
    If we exclude 255s, mean will be "tissue mean".
    Background 255 will become (255 - tissue_mean) / tissue_std -> Very bright positive value.
    This is generally desired (background is "null" or "bright").
    
    Decision: Compute stats on ROI (non-255 pixels or use the mask if available).
    Since we save clean PNGs, let's allow passing masks or just checking for [255,255,255] if mask unavailable.
    Ideally we pass masks to be precise.
    """
    
    pixel_counts = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    
    for img_path in image_paths_list:
        img = np.array(Image.open(img_path).convert('RGB'))
        # We can try to detect background if it is pure white, 
        # OR we just rely on the fact we whitened it.
        # Let's start by including ALL pixels (standard ImageNet style),
        # UNLESS user specifically asked for tissue-only statistics.
        # User prompt said: "計算完整個資料集的 Mean...".
        # Usually for pre-trained models (ImageNet), we normalize by ImageNet stats.
        # For custom training, we normalize by dataset stats. 
        # If dataset has 50% white background, that's part of the dataset distribution.
        
        # However, to be more robust for sliding windows, tissue-only stats are often preferred.
        # Let's support masking if provided, else full image.
        
        # Simplification: Compute statistics of the whole image (including white background).
        # This ensures (Pixel - Mean) is minimal for "average" pixels.
        
        pixels = img.reshape(-1, 3) / 255.0 # Normalize to 0-1 for Pytorch style stats
        
        pixel_counts += pixels.shape[0]
        channel_sum += np.sum(pixels, axis=0)
        channel_sq_sum += np.sum(pixels ** 2, axis=0)
        
    if pixel_counts == 0:
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
        
    mean = channel_sum / pixel_counts
    # Variance = E[X^2] - (E[X])^2
    variance = (channel_sq_sum / pixel_counts) - (mean ** 2)
    std = np.sqrt(np.maximum(variance, 0))
    
    return mean.tolist(), std.tolist()

def apply_clahe_lab(img, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Convert to LAB, apply CLAHE to L, convert back.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    merged = cv2.merge((cl, a, b))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return result

def macenko_normalize(img, HERef, maxCRef, Io=240, alpha=1, beta=0.15):
    """
    Macenko Stain Normalization using reference stain vectors.
    
    Args:
        img: Input RGB image (uint8)
        HERef: Reference Stain Matrix (2x3)
        maxCRef: Reference Max Stain Concentrations (2,)
        
    Returns:
        Normalized image (uint8)
    """
    h, w, c = img.shape
    img = img.reshape((-1, 3))

    # Optical Density
    OD = -np.log((img.astype(np.float64) + 1) / Io)
    
    # Remove data with OD intensity less than beta
    ODhat = OD[(OD > beta).any(axis=1)]
    
    if ODhat.shape[0] < 10:
        return img.reshape((h, w, c)) # Failed to extract OD

    # SVD
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    # Project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:, 1:3])
    
    phi = np.arctan2(That[:, 1], That[:, 0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    
    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # Heuristic to order H and E vectors
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T
        
    # Rows correspond to channels (RGB), columns to stains (H, E)
    # Stain concentrations
    Y = np.reshape(OD, (-1, 3)).T
    
    # Determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]
    
    # Normalize stain concentrations
    maxC = np.percentile(C, 99, axis=1)
    # Prevent divide by zero
    maxC = np.maximum(maxC, 1e-8) 
    
    C = C / maxC[:, None]
    C = C * maxCRef[:, None]
    
    # Recreate the image using the reference stain matrix
    Inorm = io_exp(np.dot(HERef, C))
    Inorm = np.reshape(Inorm.T, (h, w, 3))
    Inorm = np.clip(Inorm, 0, 255).astype(np.uint8)
    
    return Inorm

def io_exp(x, Io=240):
    # Prevent overflow in exp. OD x should theoretically be >= 0.
    # If x is very negative, -x is huge positive -> exp explodes.
    # Io * exp(-x) should not exceed 255 significantly.
    # exp(-x) <= 255/240 ~ 1.06
    # -x <= ln(1.06) ~ 0.06
    # x >= -0.06
    # Let's be safe and clamp x to a reasonable lower bound like -10 
    # (which would result in a very bright pixel, later clipped to 255).
    # Realistically, overflow happens when x is < -700.
    x = np.maximum(x, -100) 
    return Io * np.exp(-x)

def get_stain_vectors(img, Io=240, alpha=1, beta=0.15):
    """
    Extract stain matrix and max concentrations from a single image.
    """
    h, w, c = img.shape
    img = img.reshape((-1, 3))
    
    OD = -np.log((img.astype(np.float64) + 1) / Io)
    ODhat = OD[(OD > beta).any(axis=1)]
    
    if ODhat.shape[0] < 10:
        return None, None
        
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    That = ODhat.dot(eigvecs[:, 1:3])
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T
        
    Y = np.reshape(OD, (-1, 3)).T
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]
    maxC = np.percentile(C, 99, axis=1)
    
    return HE, maxC
