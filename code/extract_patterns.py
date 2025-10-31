import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# ==========================================
# 경로 설정
# ==========================================
SEG_ROOT = "../runs/masks_final/test"   # segmentation된 얼굴 이미지
OUT_ROOT = "../runs/patterns/test_gray"      # 무늬 결과 저장
# ==========================================


def apply_CLAHE_gray(img_rgb, clip=2.0, tiles=(8,8)):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    out = clahe.apply(gray)
    return out  # uint8 grayscale



def apply_unsharp_gray(img_rgb, k=1.5, sigma=1.0):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharp = np.clip(gray * (1 + k) - blur * k, 0, 255).astype(np.uint8)
    return sharp



# def apply_sobel(gray):
#     gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
#     gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
#     mag = cv2.magnitude(gx, gy)
#     mag = (mag / (mag.max() + 1e-6) * 255).astype(np.uint8)
#     return mag


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    files = [f for f in os.listdir(SEG_ROOT) if f.endswith("_seg.png")]
    print(f"[INFO] Found {len(files)} seg images")

    for f in tqdm(files):
        path = os.path.join(SEG_ROOT, f)
        base = f.replace("_seg.png", "")
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ① CLAHE
        clahe = apply_CLAHE_gray(img_rgb)
        Image.fromarray(clahe).save(os.path.join(OUT_ROOT, f"{base}_clahe.png"))

        # ② Unsharp Mask
        sharp = apply_unsharp_gray(img_rgb)
        Image.fromarray(sharp).save(os.path.join(OUT_ROOT, f"{base}_sharp.png"))

        # ③ Sobel Edge
        # gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        # sobel = apply_sobel(gray)
        # Image.fromarray(sobel).save(os.path.join(OUT_ROOT, f"{base}_sobel.png"))

    print(f"[DONE] Pattern extraction saved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
