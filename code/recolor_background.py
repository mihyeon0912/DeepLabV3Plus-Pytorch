import os, cv2
from tqdm import tqdm

MASK_FINAL_ROOT = "../runs/masks_final"   # train / test 폴더 포함
OUT_ROOT        = "../runs/green_bg"
GREEN_BGR       = (0, 255, 0)             # 배경 색상 (B,G,R)

def recolor_black_to_green(bgr):
    bg = (bgr[:,:,0]==0) & (bgr[:,:,1]==0) & (bgr[:,:,2]==0)
    out = bgr.copy()
    out[bg] = GREEN_BGR
    return out

def process(split):
    src = os.path.join(MASK_FINAL_ROOT, split)
    dst = os.path.join(OUT_ROOT, split)
    os.makedirs(dst, exist_ok=True)

    # *_seg.png 만!
    files = [f for f in os.listdir(src)
             if f.endswith(".png") and f.endswith("_seg.png")]
    print(f"[INFO] {split}: {len(files)} files (*_seg.png)")

    for f in tqdm(files):
        inp = os.path.join(src, f)
        img = cv2.imread(inp)
        if img is None:
            print(f"[WARN] cannot read: {inp}")
            continue
        out = recolor_black_to_green(img)
        cv2.imwrite(os.path.join(dst, f.replace("_seg.png", "_seg_green.png")), out)

    print(f"[DONE] saved to {dst}")

def main():
    for split in ["train", "test"]:
        if os.path.isdir(os.path.join(MASK_FINAL_ROOT, split)):
            process(split)
        else:
            print(f"[SKIP] no folder: {split}")

if __name__ == "__main__":
    main()
