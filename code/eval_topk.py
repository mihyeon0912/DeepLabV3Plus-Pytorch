"""
실행예시:
python3 code/eval_topk.py \
  --rgb_root ./head \
  --bw_root ./head/contrast \
  --test_json ./head/test.json \
  --ckpt ./experiments/best_state.pth \
  --csv_out ./experiments/topk_results.csv
"""


import os, json, argparse
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# 우리 코드의 모델 (masked-attention fusion 버전)
from model import CombinedEmbedding

# ======== 기본 경로 ========
RGB_ROOT_DEF = "/home/work/cow_segmentation/DeepLabV3Plus-Pytorch/head"
BW_ROOT_DEF  = "/home/work/cow_segmentation/DeepLabV3Plus-Pytorch/head/contrast"
CKPT_DEF     = "/home/work/cow_segmentation/DeepLabV3Plus-Pytorch/experiments/best_state.pth"
TEST_JSON_DEF= "/home/work/cow_segmentation/DeepLabV3Plus-Pytorch/head/test.json"

# ======== 전처리 (dataset.py와 동일) ========
rgb_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
bw_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),  # (1,H,W) in [0,1]
])

def to_bw_name(rgb_fname: str) -> str:
    stem, _ = os.path.splitext(rgb_fname)  # 'xxxx-yy'
    return stem + "_seg_green_bw.png"

@torch.no_grad()
def load_image_pair(rgb_root: str, bw_root: str, split: str, fname: str, device: str):
    # RGB
    p_rgb = os.path.join(rgb_root, split, fname)
    if not os.path.isfile(p_rgb):
        raise FileNotFoundError(p_rgb)
    rgb = rgb_tf(Image.open(p_rgb).convert("RGB")).unsqueeze(0).to(device)

    # BW (없으면 0-텐서)
    p_bw = os.path.join(bw_root, split, to_bw_name(fname))
    if os.path.isfile(p_bw):
        bw = bw_tf(Image.open(p_bw).convert("L")).unsqueeze(0).to(device)
    else:
        bw = torch.zeros(1,1,224,224, device=device)
    return rgb, bw

@torch.no_grad()
def build_embeddings(model, rgb_root: str, bw_root: str, test_map: Dict[str, List[str]], device: str):
    gallery_feats, gallery_ids, gallery_names = [], [], []
    query_feats,   query_ids,   query_names   = [], [], []

    for cid, files in test_map.items():
        if len(files) == 0:
            continue
        # 갤러리 1장, 쿼리는 나머지
        g_name  = files[0]
        q_names = files[1:] if len(files) > 1 else []

        # gallery
        rgb, bw = load_image_pair(rgb_root, bw_root, "test", g_name, device)
        g_emb = model(rgb, bw)    # (1,D)
        gallery_feats.append(g_emb.cpu())
        gallery_ids.append(cid)
        gallery_names.append(g_name)

        # queries
        for qn in q_names:
            rgb, bw = load_image_pair(rgb_root, bw_root, "test", qn, device)
            q_emb = model(rgb, bw)
            query_feats.append(q_emb.cpu())
            query_ids.append(cid)
            query_names.append(qn)

    if len(query_feats) == 0:
        # 모든 ID가 1장뿐이면 쿼리가 없음 → 갤러리 일부를 쿼리로 분할
        n = len(gallery_feats)
        half = max(1, n//2)
        query_feats   = gallery_feats[:half]
        query_ids     = gallery_ids[:half]
        query_names   = gallery_names[:half]
        gallery_feats = gallery_feats[half:]
        gallery_ids   = gallery_ids[half:]
        gallery_names = gallery_names[half:]

    G = torch.cat(gallery_feats, dim=0)   # (Ng, D)
    Q = torch.cat(query_feats,   dim=0)   # (Nq, D)
    return (G, gallery_ids, gallery_names), (Q, query_ids, query_names)

@torch.no_grad()
def evaluate_topk(G, gid, gname, Q, qid, qname, ks=(1,3,5)):
    # cosine similarity
    S = F.linear(F.normalize(Q), F.normalize(G))  # (Nq, Ng)
    S_np = S.cpu().numpy()

    # 랭킹
    idx = np.argsort(-S_np, axis=1)  # 내림차순 인덱스
    gid_arr = np.array(gid)

    results = {}
    for k in ks:
        topk_ids = gid_arr[idx[:, :k]]
        hit = (topk_ids == np.array(qid)[:, None]).any(axis=1).mean()
        results[f"top{k}"] = float(hit)

    return results, idx, S_np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_root",  default=RGB_ROOT_DEF)
    ap.add_argument("--bw_root",   default=BW_ROOT_DEF)
    ap.add_argument("--test_json", default=TEST_JSON_DEF)
    ap.add_argument("--ckpt",      default=CKPT_DEF,
                    help="best_state.pth(권장) 또는 best.ckpt(딕셔너리) 경로")
    ap.add_argument("--csv_out",   default="",
                    help="쿼리별 top5 결과를 CSV로 저장 (옵션)")
    ap.add_argument("--device",    default="cuda")
    args = ap.parse_args()

    assert os.path.isfile(args.test_json), f"not found: {args.test_json}"
    with open(args.test_json, "r", encoding="utf-8") as f:
        test_map = json.load(f)

    device = args.device if torch.cuda.is_available() and args.device=="cuda" else "cpu"

    # 새로운 fusion 모델 구성 (학습 시와 동일 하이퍼)
    model = CombinedEmbedding(
        d=256,          # attention 채널 크기
        fused_dim=128,  # 최종 임베딩 차원
        nblocks=2,      # Fusion block 개수
        nheads=4,       # Multi-head attention head 수
        pdrop=0.1       # dropout 비율
    ).to(device)

    # checkpoint 로드 (best_state.pth 또는 best.ckpt 모두 지원)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state if 'model' not in state else state['model'])
    model.eval()

    # 임베딩 생성
    (G, gid, gname), (Q, qid, qname) = build_embeddings(
        model, args.rgb_root, args.bw_root, test_map, device
    )

    # 평가
    metrics, idx, S_np = evaluate_topk(G, gid, gname, Q, qid, qname, ks=(1,3))
    print(f"[EVAL] N_gallery={G.shape[0]}, N_query={Q.shape[0]}")
    print(f"[EVAL] Top-1={metrics['top1']:.4f}  Top-3={metrics['top3']:.4f}")

    # (옵션) CSV 저장
    if args.csv_out:
        import csv
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query", "qid", "rank", "gallery", "gid", "score"])
            for qi in range(Q.shape[0]):
                ranks = idx[qi, :5]
                for r, gi in enumerate(ranks, start=1):
                    w.writerow([qname[qi], qid[qi], r, gname[gi], gid[gi], float(S_np[qi, gi])])
        print(f"[EVAL] saved csv: {args.csv_out}")

if __name__ == "__main__":
    main()