import os, cv2, numpy as np
from tqdm import tqdm

# ====================== 경로/파라미터 ======================
SRC_ROOTS = [
    "../runs/green_bg/train",
    "../runs/green_bg/test",
    "../runs/masks_final/train",  # fallback (_seg.png)
    "../runs/masks_final/test",   # fallback (_seg.png)
]
OUT_ROOT  = "../runs/patterns_bw"   # 최종 흑/백 무늬 저장 폴더

USE_MASK_IF_AVAILABLE = True        # *_mask.png 있으면 우선 사용
CLAHE_CLIP = 40.0                   # CLAHE 대비 강화 정도(최대 대조비)
CLAHE_TILE = (4, 4)                 # 더 작은 타일로 세밀한 대조비 향상
UNSHARP_K  = 3.0                    # 언샤프 강도를 크게 증가
UNSHARP_SIGMA = 0.5                 # 더 강한 샤프닝을 위해 감소
BIN_METHOD = "otsu"                 # "otsu" | "adaptive"
ADAPT_BLOCK = 31                    # adaptive 시 블록 크기(홀수)
ADAPT_C     = 5                     # adaptive 시 보정값
# ==========================================================


def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_gray_keep_fg(path_img, path_mask=None):
    """
    입력 이미지에서 얼굴(전경)만 그레이로 반환하고, 배경은 0으로 둠.
    - path_img: *_seg_green.png (권장) 또는 *_seg.png
    - path_mask: *_mask.png 가 있으면 전경 마스크로 사용
    """
    img = cv2.imread(path_img, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    # RGBA → RGB
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]

    # 전경 마스크 결정
    fg = None
    if USE_MASK_IF_AVAILABLE and path_mask and os.path.isfile(path_mask):
        m = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape != (h, w): m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            fg = (m > 0).astype(np.uint8)

    if fg is None:
        # 마스크 없으면: green_bg 기준 → 초록 배경(0,255,0) 근접 픽셀은 배경으로
        b,g,r = cv2.split(img)
        green_bg = (np.abs(b-0)<=8) & (np.abs(g-255)<=8) & (np.abs(r-0)<=8)
        black_bg = (b<8) & (g<8) & (r<8)  # seg.png fallback 대비
        fg = (~(green_bg | black_bg)).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_fg = (gray * fg).astype(np.uint8)  # 배경은 0으로
    return gray_fg, fg


def clahe(gray):
    cla = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    return cla.apply(gray)

def unsharp(gray):
    if UNSHARP_K <= 0: return gray
    blur = cv2.GaussianBlur(gray, (0,0), UNSHARP_SIGMA)
    sharp = np.clip(gray*(1+UNSHARP_K) - blur*UNSHARP_K, 0, 255).astype(np.uint8)
    return sharp

def binarize(gray):
    if BIN_METHOD == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, ADAPT_BLOCK, ADAPT_C
        )
    else:
        # Otsu
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return bw

def process_dir(src_dir):
    if not os.path.isdir(src_dir): 
        print(f"[SKIP] {src_dir}")
        return

    rel = os.path.basename(os.path.dirname(src_dir)), os.path.basename(src_dir)
    out_dir = os.path.join(OUT_ROOT, *[p for p in rel if p])
    ensure_dir(out_dir)

    # 우선순위: *_seg_green.png → 없으면 *_seg.png
    all_files = os.listdir(src_dir)
    seg_green = [f for f in all_files if f.endswith("_seg_green.png")]
    seg_plain = [f for f in all_files if f.endswith("_seg.png") and not f.endswith("_seg_green.png")]

    targets = seg_green if seg_green else seg_plain
    print(f"[INFO] {src_dir}: {len(targets)} targets")

    for f in tqdm(targets):
        base = f.replace("_seg_green.png", "").replace("_seg.png", "")
        p_img = os.path.join(src_dir, f)

        # mask 경로 후보
        p_mask = os.path.join(src_dir, base + "_mask.png")
        gray_fg, fg = load_gray_keep_fg(p_img, p_mask if os.path.isfile(p_mask) else None)
        if gray_fg is None: 
            print(f"[WARN] read fail: {p_img}"); 
            continue

        # 1) CLAHE → 2) Unsharp → 3) 이진화(흑/백)
        g1 = clahe(gray_fg)
        g2 = unsharp(g1)
        bw = binarize(g2)

        # 전경만 남기고 배경은 0(검정) 고정
        bw_fg = (bw * (fg>0)).astype(np.uint8)

        # 저장
        cv2.imwrite(os.path.join(out_dir, base + "_clahe.png"), g1)
        cv2.imwrite(os.path.join(out_dir, base + "_sharp.png"), g2)
        cv2.imwrite(os.path.join(out_dir, base + "_bw.png"), bw_fg)

def main():
    for d in SRC_ROOTS:
        process_dir(d)

if __name__ == "__main__":
    main()
