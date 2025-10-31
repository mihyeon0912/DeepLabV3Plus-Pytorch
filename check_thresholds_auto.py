import os, argparse, cv2, numpy as np, matplotlib
matplotlib.use("Agg")  # 서버 환경: 창 없이 저장만
import matplotlib.pyplot as plt

IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
PROB_SUFFIXES = ("_cowprob.npy", "_cowprob_8u.png")

def load_prob(prob_path):
    if prob_path.endswith(".npy"):
        prob = np.load(prob_path).astype(np.float32)
        return np.clip(prob, 0, 1)
    p8 = cv2.imread(prob_path, cv2.IMREAD_GRAYSCALE)
    if p8 is None:
        raise RuntimeError(f"Failed to read prob as image: {prob_path}")
    return (p8.astype(np.float32) / 255.0)

def find_image(img_root, rel_dir, base):
    for ext in IMG_EXTS:
        cand = os.path.join(img_root, rel_dir, base + ext)
        if os.path.isfile(cand):
            return cand
    return None

def overlay_mask(img_rgb, mask, color=(255,0,0), alpha=0.45):
    out = img_rgb.copy()
    color_arr = np.zeros_like(img_rgb)
    color_arr[mask > 0] = color
    out = (out * (1 - alpha) + color_arr * alpha).astype(np.uint8)
    return out

def make_grid(original, overlays, titles, save_path):
    cols = len(overlays) + 1
    plt.figure(figsize=(4*cols, 4))
    plt.subplot(1, cols, 1); plt.imshow(original); plt.title("Original"); plt.axis("off")
    for i, (ov, t) in enumerate(zip(overlays, titles), start=2):
        plt.subplot(1, cols, i); plt.imshow(ov); plt.title(t); plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob_root", required=True)
    ap.add_argument("--img_root",  required=True)
    ap.add_argument("--out_root",  required=True)
    ap.add_argument("--thresholds", type=str, default="0.30,0.40,0.50,0.60,0.70")
    args = ap.parse_args()

    thr_list = [float(x) for x in args.thresholds.split(",")]
    os.makedirs(args.out_root, exist_ok=True)

    for root, _, files in os.walk(args.prob_root):
        rel = os.path.relpath(root, args.prob_root)
        out_dir = os.path.join(args.out_root, rel)
        os.makedirs(out_dir, exist_ok=True)

        prob_files = []
        for f in files:
            for suf in PROB_SUFFIXES:
                if f.endswith(suf):
                    prob_files.append((f, suf))
                    break
        if not prob_files:
            continue

        for f, suf in prob_files:
            prob_path = os.path.join(root, f)
            base = f[:-len(suf)]

            img_path = find_image(args.img_root, rel, base)
            prob = load_prob(prob_path)
            H, W = prob.shape

            if img_path is not None:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    original = np.stack([prob*255]*3, axis=2).astype(np.uint8)
                else:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    if img_rgb.shape[:2] != (H, W):
                        img_rgb = cv2.resize(img_rgb, (W, H))
                    original = img_rgb
            else:
                original = np.stack([prob*255]*3, axis=2).astype(np.uint8)

            overlays, titles = [], []
            for thr in thr_list:
                mask = (prob >= thr).astype(np.uint8)
                cow_ratio = mask.mean()
                ov = overlay_mask(original, mask, color=(255,0,0), alpha=0.45)
                overlays.append(ov)
                titles.append(f"thr={thr:.2f}\nratio={cow_ratio:.3f}")

            save_path = os.path.join(out_dir, f"{base}_thresholds.png")
            make_grid(original, overlays, titles, save_path)
            print(f"[SAVE] {save_path}")

if __name__ == "__main__":
    main()
