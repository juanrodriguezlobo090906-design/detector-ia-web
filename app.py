import cv2
import numpy as np
from PIL import Image

def fft_score(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1)
    # normalización simple
    score = np.mean(mag) / (np.std(mag) + 1e-6)
    score = min(max(score / 10, 0), 1)
    return score

def texture_score(gray):
    # detector de bordes + varianza de textura
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.sum(edges > 0) / edges.size
    var = np.var(gray) / (255.0**2)
    score = (edge_ratio * 0.6 + var * 0.4)
    score = min(max(score, 0), 1)
    return score

def patch_analysis(gray):
    h, w = gray.shape
    size = 128
    scores = []
    for y in range(0, max(h - size, 1), size):
        for x in range(0, max(w - size, 1), size):
            patch = gray[y:y+size, x:x+size]
            if patch.size == 0:
                continue
            scores.append(texture_score(patch))
    if not scores:
        return 0.5
    return float(np.mean(scores))

def analyze_image(path):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    s_fft = fft_score(gray)
    s_tex = texture_score(gray)
    s_patch = patch_analysis(gray)

    # combinación calibrada
    final = (s_fft * 0.4) + (s_tex * 0.3) + (s_patch * 0.3)

    return round(final * 100, 2)