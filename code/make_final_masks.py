import os, cv2, numpy as np
from PIL import Image
from tqdm import tqdm

# ============================
# 고정 경로 & 파라미터
# ============================
PROB_ROOT = "../runs/cowprob_tv/train"   # *_cowprob.npy 가 저장된 폴더
IMG_ROOT  = "../head/train"              # 원본 이미지 폴더
OUT_ROOT  = "../runs/masks_final/train"  # 최종 마스크/오버레이 저장 폴더

THRESH = 0.30           # 확률 임계값(고정)
MIN_AREA_RATIO = 0.01   # 이미지 대비 최소 면적 필터(1% 미만은 제거)
CLOSE_K = 7             # morphology close 커널 크기(홀 메우기)
# ============================


def load_prob(prob_path):
    prob = np.load(prob_path).astype(np.float32)  # (H,W) 0~1
    return np.clip(prob, 0, 1)


def binarize(prob, thr=THRESH):
    return (prob >= thr).astype(np.uint8)


def fill_holes(mask):
    # flood fill로 내부 구멍 메우기
    h, w = mask.shape
    ff = mask.copy()
    cv2.floodFill(ff, np.zeros((h+2, w+2), np.uint8), (0, 0), 255)
    inv = cv2.bitwise_not(ff)
    return cv2.bitwise_or(mask*255, inv) // 255


def morph_close(mask, k=CLOSE_K):
    if k <= 0: return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def remove_small(mask, min_area):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, num):  # 0은 배경
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    return keep


def make_center_weight(h, w, sigma_ratio=0.35):
    # 중앙에 가까울수록 가중치 ↑ (얼굴이 대체로 중앙에 있다는 약한 prior)
    cy, cx = h/2.0, w/2.0
    y, x = np.mgrid[0:h, 0:w]
    d2 = (x-cx)**2 + (y-cy)**2
    sigma2 = (sigma_ratio*max(h, w))**2
    wmap = np.exp(-d2/(2*sigma2))
    wmap /= (wmap.max() + 1e-6)
    return wmap.astype(np.float32)


def select_component_centered(mask):
    """
    여러 컴포넌트가 있을 때 '가운데에 가까우며 크기도 적당히 큰' 것을 선택.
    점수 = 0.5*정규화 면적 + 0.5*중심 가중치 평균
    """
    h, w = mask.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 2:  # 배경 + 1개 → 그대로 반환
        return mask

    wmap = make_center_weight(h, w)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)  # index shift
    areas_norm = areas / (h*w)

    best_i, best_score = None, -1
    for i in range(1, num):
        comp = (labels == i).astype(np.uint8)
        area_n = areas_norm[i-1]
        center_mean = float((wmap * comp).sum() / (comp.sum() + 1e-6))
        score = 0.5*area_n + 0.5*center_mean
        if score > best_score:
            best_score, best_i = score, i

    out = (labels == best_i).astype(np.uint8)
    return out


def overlay(img_rgb, mask01, alpha=0.45):
    color = np.zeros_like(img_rgb)
    color[mask01 > 0] = [255, 0, 0]
    return (img_rgb*(1-alpha) + color*alpha).astype(np.uint8)


def process_one(prob_path, img_path, save_base):
    prob = load_prob(prob_path)
    H, W = prob.shape

    # 1) 임계값
    m = binarize(prob, THRESH)

    # 2) 후처리: close → 구멍 메우기 → 소형 제거
    m = morph_close(m, CLOSE_K)
    m = fill_holes(m)
    m = remove_small(m, int(MIN_AREA_RATIO * H * W))

    # 3) 중앙성 기준 컴포넌트 선택(얼굴만 남기기 시도)
    m = select_component_centered(m)

    # 4) 저장: 최종 마스크 / 오버레이 / 알파크롭
    os.makedirs(os.path.dirname(save_base), exist_ok=True)
    Image.fromarray((m*255).astype(np.uint8)).save(save_base + "_mask.png")

    img_bgr = cv2.imread(img_path)
    if img_bgr is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[:2] != (H, W):
            img_rgb = cv2.resize(img_rgb, (W, H))
        Image.fromarray(overlay(img_rgb, m)).save(save_base + "_overlay.png")

        # 배경을 검정으로 제거한 결과(얼굴만 출력)
        seg = (img_rgb * np.repeat(m[..., None], 3, axis=2)).astype(np.uint8)
        Image.fromarray(seg).save(save_base + "_seg.png")


def main():
    print(f"[INFO] THRESH={THRESH}  prob_root={PROB_ROOT}")
    for root, _, files in os.walk(PROB_ROOT):
        rel = os.path.relpath(root, PROB_ROOT)
        for f in tqdm(files, desc=f"[{rel}]"):
            if not f.endswith("_cowprob.npy"):
                continue
            base = f[:-len("_cowprob.npy")]
            prob_p = os.path.join(root, f)

            # 원본 이미지 찾기
            img_p = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                cand = os.path.join(IMG_ROOT, rel, base + ext)
                if os.path.isfile(cand):
                    img_p = cand; break
            if img_p is None:
                print(f"[WARN] 원본 이미지 없음: {base}")
                continue

            save_base = os.path.join(OUT_ROOT, rel, base)
            process_one(prob_p, img_p, save_base)

    print(f"[DONE] 결과 저장: {OUT_ROOT}")


if __name__ == "__main__":
    main()
