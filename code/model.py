import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# =========================================================
# 기존 이미지 임베딩(ResNet50 → 128-d)
# =========================================================
class ResNet50Embedding(nn.Module):
    def __init__(self, embed_dim: int = 128, pretrained: bool = True):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        in_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.fc = nn.Linear(in_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: (B,3,H,W)
        f = self.backbone(x)          # (B,in_dim)
        e = self.fc(f)                # (B,embed_dim)
        e = F.normalize(e, dim=1)     # L2 normalize
        return e


# =========================================================
# 패턴 BW 임베딩(2층 CNN → 128-d)
# =========================================================
class PatternCNN(nn.Module):
    def __init__(self, embed_dim: int = 128, pretrained: bool = True):
        super().__init__()
        # 1) resnet18 불러오기
        self.backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # 2) 첫 conv를 1채널용으로 교체 (pretrained 가중치 평균으로 초기화)
        old_conv = self.backbone.conv1  # (in=3, out=64, k7,s2,p3)
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        with torch.no_grad():
            if pretrained:
                # RGB 가중치 평균값으로 초기화 → 채널 수만 1로
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        self.backbone.conv1 = new_conv

        # 3) FC를 제거하고 (global avgpool → 512), 임베딩 투영 레이어 추가
        in_feat = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()
        self.proj = nn.Linear(in_feat, embed_dim)

    def forward(self, x_bw: torch.Tensor) -> torch.Tensor:
        """
        x_bw: (B,1,H,W), 값 범위는 [0,1] 또는 정규화된 범위.
        """
        feat = self.backbone(x_bw)        # (B, 512)
        emb  = self.proj(feat)            # (B, embed_dim)
        emb  = F.normalize(emb, dim=1)    # L2 정규화
        return emb


# =========================================================
# 결합 임베딩: RGB(128) + BW(128) → 128
# =========================================================
class CombinedEmbedding(nn.Module):
    def __init__(self, img_embedder: nn.Module, pat_dim: int = 128, fused_dim: int = 128):
        super().__init__()
        self.img = img_embedder
        self.pat = PatternCNN(out_dim=pat_dim)

        img_dim = getattr(self.img, "embed_dim", 128)
        in_dim = img_dim + pat_dim

        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, fused_dim)
        )
        self.embed_dim = fused_dim

    def forward(self, img_rgb, bw_1ch):
        e_img = self.img(img_rgb)      # (B,128)
        e_pat = self.pat(bw_1ch)       # (B,128) or pat_dim
        e = torch.cat([e_img, e_pat], dim=1)
        e = self.proj(e)               # (B,fused_dim)
        e = F.normalize(e, dim=1)
        return e