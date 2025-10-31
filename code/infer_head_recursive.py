# infer_head_recursive.py
# ------------------------------------------------------------
# DeepLabV3+ (VainF/DeepLabV3Plus-Pytorch) 사전학습 모델로
# 입력 루트(예: ./head) 하위의 모든 이미지 폴더를 재귀적으로 순회하며
# 인덱스 마스크와 컬러 오버레이를 저장합니다.
# ------------------------------------------------------------
import os
import argparse
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def build_colormap(k: int) -> np.ndarray:
    """간단한 랜덤 컬러맵(배경=검정)."""
    np.random.seed(0)
    cmap = np.zeros((k, 3), np.uint8)
    cmap[0] = [0, 0, 0]
    for i in range(1, k):
        cmap[i] = np.random.randint(0, 255, size=3, dtype=np.uint8)
    return cmap


def overlay(img_rgb: np.ndarray, mask: np.ndarray, cmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """원본 RGB 위에 인덱스 마스크를 컬러로 오버레이."""
    color = cmap[mask]
    out = (img_rgb * (1 - alpha) + color * alpha).astype(np.uint8)
    return out


def preprocess(img_rgb: np.ndarray, long_side: int):
    """
    - 긴 변을 long_side로 리사이즈
    - 32 배수 패딩(DeepLab 안정)
    - ImageNet 정규화
    """
    h, w = img_rgb.shape[:2]
    scale = long_side / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

    ph = (32 - nh % 32) % 32
    pw = (32 - nw % 32) % 32
    padded = cv2.copyMakeBorder(resized, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    x = torch.from_numpy(padded).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
    x = ((x - mean) / std).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W

    meta = {"orig": (h, w), "resized": (nh, nw), "pad": (ph, pw)}
    return x, meta


@torch.no_grad()
def infer(model, x: torch.Tensor, device: torch.device) -> np.ndarray:
    x = x.to(device)
    out = model(x)
    if isinstance(out, dict) and "out" in out:
        out = out["out"]
    pred = torch.argmax(out[0], dim=0).cpu().numpy().astype(np.uint8)
    return pred


def depad_resize(mask: np.ndarray, meta: dict) -> np.ndarray:
    nh, nw = meta["resized"]
    ph, pw = meta["pad"]
    oh, ow = meta["orig"]
    mask = mask[:nh, :nw]
    mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
    return mask


def load_model(backbone: str, device: torch.device, repo_source: str, repo_dir: str):
    """
    1) source='local'이면 로컬 경로의 hubconf.py 사용
    2) 실패하면 원격 'VainF/DeepLabV3Plus-Pytorch:main'으로 재시도
    """
    model = None
    err_local = None

    if repo_source == "local":
        try:
            if not os.path.isfile(os.path.join(repo_dir, "hubconf.py")):
                raise FileNotFoundError(f"hubconf.py not found in {repo_dir}")
            model = torch.hub.load(
                repo_dir,
                f"deeplabv3plus_{backbone}",
                pretrained=True,
                source="local",
                trust_repo=True,
            )
        except Exception as e:
            err_local = e
            print(f"[WARN] Local hub load failed: {e}\n→ Falling back to remote repo...")

    if model is None:
        # 원격(main 브랜치 명시)
        model = torch.hub.load(
            "VainF/DeepLabV3Plus-Pytorch:main",
            f"deeplabv3plus_{backbone}",
            pretrained=True,
            trust_repo=True,
        )

    model = model.to(device).eval()

    # 클래스 수 파악
    try:
        num_classes = model.classifier.classifier[-1].out_channels
    except Exception as e:
        raise RuntimeError(
            "Unexpected classifier structure; check the repository version."
        ) from e

    return model, num_classes, err_local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True, help="예: ./head")
    ap.add_argument("--output_root", required=True, help="예: ./runs/preds_head")
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet101", "mobilenet"])
    ap.add_argument("--long_side", type=int, default=640)
    ap.add_argument("--device", default="cuda", help="'cuda' 또는 'cpu'")
    ap.add_argument("--save_overlay", action="store_true", help="오버레이 이미지도 저장")
    ap.add_argument("--repo_source", default="local", choices=["local", "remote"],
                    help="먼저 시도할 소스(local/remote). local 실패 시 remote로 자동 fallback")
    ap.add_argument("--repo_dir", default=os.path.abspath("."), help="로컬 레포 경로(예: /path/to/DeepLabV3Plus-Pytorch)")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # 모델 로드
    model, num_classes, _ = load_model(args.backbone, device, args.repo_source, args.repo_dir)
    cmap = build_colormap(num_classes)

    # 재귀 순회
    for root, _, files in os.walk(args.input_root):
        rel = os.path.relpath(root, args.input_root)
        out_dir = os.path.join(args.output_root, rel)
        ov_dir = os.path.join(out_dir, "overlay")
        os.makedirs(out_dir, exist_ok=True)
        if args.save_overlay:
            os.makedirs(ov_dir, exist_ok=True)

        images = [f for f in files if f.lower().endswith(EXTS)]
        if not images:
            continue

        for f in tqdm(images, desc=f"[{rel}]"):
            in_p = os.path.join(root, f)
            img_bgr = cv2.imread(in_p)
            if img_bgr is None:
                continue
            img = img_bgr[:, :, ::-1]  # BGR->RGB

            x, meta = preprocess(img, args.long_side)
            pred = infer(model, x, device)
            pred = depad_resize(pred, meta)

            base, _ = os.path.splitext(f)
            # 인덱스 마스크 저장
            Image.fromarray(pred).save(os.path.join(out_dir, f"{base}_mask.png"))

            # 오버레이 저장(옵션)
            if args.save_overlay:
                ov = overlay(img, pred, cmap, alpha=0.5)
                Image.fromarray(ov).save(os.path.join(ov_dir, f"{base}_overlay.png"))


if __name__ == "__main__":
    main()
