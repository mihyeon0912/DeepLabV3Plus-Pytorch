import os, random, math, time, argparse
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset import TripletDataset, PairDataset
from model import ResNet50Embedding, CombinedEmbedding


# =========================================================
# [1] 데이터 루트 고정 경로 설정
# =========================================================
RGB_ROOT = "/home/work/cow_segmentation/DeepLabV3Plus-Pytorch/head"
BW_ROOT  = "/home/work/cow_segmentation/DeepLabV3Plus-Pytorch/head/contrast"

# =========================================================
# [2] 디버깅용 경로 출력
# =========================================================
print("[DEBUG] Current Working Directory :", os.getcwd())
print("[DEBUG] Fixed RGB Root Path        :", RGB_ROOT)
print("[DEBUG] Fixed BW Root Path         :", BW_ROOT)
# =========================================================


# =========================================================
# 트리플릿 손실
# =========================================================
class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, a, p, n):
        d_ap = 1 - torch.sum(a * p, dim=1)
        d_an = 1 - torch.sum(a * n, dim=1)
        loss = torch.clamp(d_ap - d_an + self.margin, min=0).mean()
        return loss


# =========================================================
# 유틸
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def cosine_sim(x, y):
    return torch.sum(x * y, dim=1)


# =========================================================
# 검증 루프 (PairDataset)
# =========================================================
@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    sims = []
    labels = []
    for (img1, img2, lab, bw1, bw2) in loader:
        img1 = img1.to(device); img2 = img2.to(device)
        bw1  = bw1.to(device) if bw1 is not None else torch.zeros(img1.size(0),1,img1.size(2),img1.size(3), device=device)
        bw2  = bw2.to(device) if bw2 is not None else torch.zeros(img2.size(0),1,img2.size(2),img2.size(3), device=device)

        e1 = model(img1, bw1)
        e2 = model(img2, bw2)
        s  = cosine_sim(e1, e2).cpu().numpy()
        sims.append(s)
        labels.append(lab.numpy())
    sims = np.concatenate(sims)
    labels = np.concatenate(labels)

    # threshold sweep
    thr_space = np.linspace(-1.0, 1.0, 401)
    best_acc, best_thr = 0.0, 0.0
    for thr in thr_space:
        pred = (sims >= thr).astype(np.int32)
        acc = (pred == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_acc, best_thr


# =========================================================
# 메인
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    # =========================================================
    # 고정 경로 버전 (root 입력받지 않음)
    # =========================================================
    ap.add_argument("--rgb_root", default=RGB_ROOT)
    ap.add_argument("--bw_root",  default=BW_ROOT)
    # =========================================================
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epoch_len", type=int, default=20000)
    ap.add_argument("--val_pairs", type=int, default=4000)
    ap.add_argument("--pos_ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="./experiments")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================
    # 데이터셋 로드
    # =========================================================
    train_json = os.path.join(args.rgb_root, "train.json")
    test_json  = os.path.join(args.rgb_root, "test.json")
    assert os.path.isfile(train_json) and os.path.isfile(test_json), \
        f"JSON 파일이 존재하지 않습니다: {train_json}, {test_json}"

    train_ds = TripletDataset(
        root=args.rgb_root,
        json_path=train_json,
        train=True,
        epoch_len=args.epoch_len,
        bw_root=args.bw_root
    )
    val_ds = PairDataset(
        root=args.rgb_root,
        json_path=test_json,
        bw_root=args.bw_root,
        num_pairs=args.val_pairs,
        pos_ratio=args.pos_ratio
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # =========================================================
    # 모델/손실/옵티마이저 설정
    # =========================================================
    # old:
    # img_embedder = ResNet50Embedding(embed_dim=128, pretrained=True)
    # model = CombinedEmbedding(img_embedder=img_embedder, pat_dim=128, fused_dim=128).to(device)

    # new:
    model = CombinedEmbedding(d=256, fused_dim=128, nblocks=2, nheads=4, pdrop=0.1).to(device)
    
    criterion = TripletLoss(margin=args.margin)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs
    )
    scaler = GradScaler()

    best_acc = 0.0
    best_thr = 0.0
    ckpt_path = os.path.join(args.out, "best.ckpt")

    # =========================================================
    # 학습 루프
    # =========================================================
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for (a_rgb, p_rgb, n_rgb, a_bw, p_bw, n_bw) in train_loader:
            a_rgb = a_rgb.to(device); p_rgb = p_rgb.to(device); n_rgb = n_rgb.to(device)

            # BW가 None이면 dummy zero tensor 생성
            if a_bw is None:
                a_bw = torch.zeros(a_rgb.size(0), 1, 224, 224, device=device)
                p_bw = torch.zeros_like(a_bw)
                n_bw = torch.zeros_like(a_bw)
            else:
                a_bw = a_bw.to(device); p_bw = p_bw.to(device); n_bw = n_bw.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                a_emb = model(a_rgb, a_bw)
                p_emb = model(p_rgb, p_bw)
                n_emb = model(n_rgb, n_bw)
                loss = criterion(a_emb, p_emb, n_emb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item()

        avg_loss = running / max(1, len(train_loader))

        # =========================================================
        # 검증
        # =========================================================
        val_acc, val_thr = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch:03d}] loss={avg_loss:.4f} | val_acc={val_acc:.4f} | thr={val_thr:.3f}")

        # 최고 성능 갱신 시 저장
        # ... (검증 후 최고 성능 갱신 시)
    if val_acc > best_acc:
        best_acc, best_thr = val_acc, val_thr
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "best_acc": best_acc,
            "best_thr": best_thr,
        }
        torch.save(payload, ckpt_path)  # 기존 best.ckpt

        # 추가: state_dict(.pth), full model(.pth)
        torch.save(model.state_dict(), os.path.join(args.out, "best_state.pth"))
        torch.save(model,              os.path.join(args.out, "best_model.pth"))
        print(f"  -> best updated. acc={best_acc:.4f}, thr={best_thr:.3f}")
        print(f"     saved: {ckpt_path}, best_state.pth, best_model.pth")
    print(f"[DONE] best_acc={best_acc:.4f}, best_thr={best_thr:.3f}, ckpt={ckpt_path}")


if __name__ == "__main__":
    main()