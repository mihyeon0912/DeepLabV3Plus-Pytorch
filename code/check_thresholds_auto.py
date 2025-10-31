import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# 설정 (경로는 네 구조에 맞춰 직접 수정)
# ============================================================
PROB_ROOT = "../runs/cowprob_tv/train"   # cow 확률맵(.npy) 폴더
IMG_ROOT  = "../head/train"              # 원본 이미지 폴더
OUT_ROOT  = "../runs/compare_thresholds/train"  # 출력 저장 폴더

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]   # 테스트할 threshold 값
ALPHA = 0.45  # overlay 투명도

# ============================================================


def overlay_mask(img_rgb, mask, alpha=ALPHA):
    """마스크를 반투명하게 오버레이한 RGB 이미지 반환"""
    color = np.zeros_like(img_rgb)
    color[mask > 0] = [255, 0, 0]  # red overlay
    blended = (img_rgb * (1 - alpha) + color * alpha).astype(np.uint8)
    return blended


def save_threshold_grid(prob_path, img_path, save_path, thresholds=THRESHOLDS):
    """하나의 이미지에 대해 여러 threshold 적용 결과를 시각화"""
    prob = np.load(prob_path).astype(np.float32)  # (H, W)
    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        print(f"[WARN] 이미지 로드 실패: {img_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img_rgb.shape[:2] != prob.shape:
        img_rgb = cv2.resize(img_rgb, (prob.shape[1], prob.shape[0]))

    cols = len(thresholds) + 1
    plt.figure(figsize=(4 * cols, 4))

    # 원본
    plt.subplot(1, cols, 1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis("off")

    # 각 threshold 적용
    for i, thr in enumerate(thresholds, start=2):
        mask = (prob >= thr).astype(np.uint8)
        over = overlay_mask(img_rgb, mask)
        ratio = mask.mean()  # foreground 비율
        plt.subplot(1, cols, i)
        plt.imshow(over)
        plt.title(f"thr={thr:.2f}\nratio={ratio:.3f}")
        plt.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"[INFO] Threshold 비교 시작")
    print(f"prob_root={PROB_ROOT}")
    print(f"img_root={IMG_ROOT}")
    print(f"out_root={OUT_ROOT}")

    for root, _, files in os.walk(PROB_ROOT):
        rel = os.path.relpath(root, PROB_ROOT)
        for f in tqdm(files, desc=f"[{rel}]"):
            if not f.endswith("_cowprob.npy"):
                continue

            base = f.replace("_cowprob.npy", "")
            prob_path = os.path.join(root, f)

            # 원본 이미지 경로 찾기
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                cand = os.path.join(IMG_ROOT, rel, base + ext)
                if os.path.isfile(cand):
                    img_path = cand
                    break

            if img_path is None:
                print(f"[WARN] 원본 이미지 없음: {base}")
                continue

            save_path = os.path.join(OUT_ROOT, rel, f"{base}_thresholds.png")
            save_threshold_grid(prob_path, img_path, save_path)

    print(f"[DONE] 결과 저장 위치 → {OUT_ROOT}")


if __name__ == "__main__":
    main()
