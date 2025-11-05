import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -------------------------------
# 0) 유틸: 버전 호환 weights 헬퍼
# -------------------------------
def _resnet50(pretrained: bool):
    try:
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    except Exception:
        return models.resnet50(pretrained=pretrained)

def _resnet18(pretrained: bool):
    try:
        return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    except Exception:
        return models.resnet18(pretrained=pretrained)


# -------------------------------
# 1) RGB/Pattern 백본: feature map 반환
# -------------------------------
class ResNet50Feature(nn.Module):
    """
    RGB용 ResNet50 백본.
    forward(x) -> 마지막 conv feature (B, 2048, H, W)  (224 입력이면 7x7)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = _resnet50(pretrained)
        # avgpool/fc 제거, layer4까지 사용
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.out_channels = 2048

    def forward(self, x):  # (B,3,H,W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)   # (B,2048,H/32,W/32)
        return x


class PatternResNet18Feature(nn.Module):
    """
    패턴(BW, 1ch)용 ResNet18 백본.
    forward(x_bw) -> 마지막 conv feature (B, 512, H, W)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = _resnet18(pretrained)
        # conv1을 1채널로 교체 (pretrained면 RGB 가중치 평균으로 초기화)
        old = base.conv1
        new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            if pretrained and old.weight.shape[1] == 3:
                new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
        base.conv1 = new

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.out_channels = 512

    def forward(self, x_bw):  # (B,1,H,W)
        x = self.stem(x_bw)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)   # (B,512,H/32,W/32)
        return x


# -------------------------------
# 2) Positional Embedding (learnable 2D)
# -------------------------------
class PosEmbedding2D(nn.Module):
    def __init__(self, d: int, H: int, W: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, H*W, d))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):  # (B, N, d)
        return x + self.pe


# -------------------------------
# 3) Fusion block: CrossAttn(Q=img, K/V=pat, masked) -> Skip
#                  SelfAttn(masked optional)         -> Skip
#                  FFN                                -> Skip
# -------------------------------
class CrossSelfFFNBlock(nn.Module):
    def __init__(self, d=256, nheads=4, ffn_mult=4, pdrop=0.1):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)

        self.cross = nn.MultiheadAttention(d, nheads, dropout=pdrop, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.drop1 = nn.Dropout(pdrop)

        self.self_attn = nn.MultiheadAttention(d, nheads, dropout=pdrop, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.drop2 = nn.Dropout(pdrop)

        self.ffn = nn.Sequential(
            nn.Linear(d, d*ffn_mult),
            nn.GELU(),
            nn.Dropout(pdrop),
            nn.Linear(d*ffn_mult, d),
        )
        self.norm3 = nn.LayerNorm(d)
        self.drop3 = nn.Dropout(pdrop)

    def forward(self, x_img, x_pat, kv_pad_mask=None, self_pad_mask=None):
        # x_img, x_pat: (B,N,d)
        # kv_pad_mask/self_pad_mask: (B,N)  True=무시될 위치
        q = self.q_proj(x_img)
        k = self.k_proj(x_pat)
        v = self.v_proj(x_pat)

        y, _ = self.cross(q, k, v, key_padding_mask=kv_pad_mask)    # Cross-Attn (배경 K/V 차단)
        x = self.norm1(x_img + self.drop1(y))

        z, _ = self.self_attn(x, x, x, key_padding_mask=self_pad_mask)  # Self-Attn (선택적 차단)
        x = self.norm2(x + self.drop2(z))

        u = self.ffn(x)
        x = self.norm3(x + self.drop3(u))
        return x


# -------------------------------
# 4) PatternFusion: proj(1x1) → pos → blocks × L → GAP → 128D
# -------------------------------
class PatternFusion(nn.Module):
    def __init__(self, c_img: int, c_pat: int, d=256, H=7, W=7, nblocks=2, nheads=4, pdrop=0.1, out_dim=128):
        super().__init__()
        self.proj_img = nn.Conv2d(c_img, d, kernel_size=1)
        self.proj_pat = nn.Conv2d(c_pat, d, kernel_size=1)
        self.pos      = PosEmbedding2D(d, H, W)
        self.blocks   = nn.ModuleList([CrossSelfFFNBlock(d, nheads, pdrop=pdrop) for _ in range(nblocks)])
        self.head     = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, out_dim))
        self.out_dim  = out_dim

    def forward(self, F_img, F_pat, fg_mask=None):
        """
        F_img: (B,Ci,H,W), F_pat: (B,Cp,h,w)  (해상도 다르면 내부에서 맞춤)
        fg_mask: (B,1,H0,W0) foreground(=1) / background(=0); 없으면 None
        """
        B, _, Hi, Wi = F_img.shape
        # 해상도 맞춤
        if F_pat.shape[-2:] != (Hi, Wi):
            F_pat = F.interpolate(F_pat, size=(Hi, Wi), mode='bilinear', align_corners=False)

        Qi = self.proj_img(F_img)               # (B,d,H,W)
        Kv = self.proj_pat(F_pat)               # (B,d,H,W)

        # 토큰화
        Qi = Qi.flatten(2).transpose(1, 2)      # (B,N,d), N=H*W
        Kv = Kv.flatten(2).transpose(1, 2)      # (B,N,d)

        # (선택) 마스크 적용
        if fg_mask is not None:
            m = F.interpolate(fg_mask, size=(Hi, Wi), mode='nearest')  # (B,1,H,W)
            m_flat = m.flatten(2).squeeze(1)                           # (B,N)
            # 배경은 0으로
            Kv = Kv * m_flat.unsqueeze(-1)
            # attention에서 배경 K/V 무시
            kv_pad_mask = (m_flat == 0)
            self_pad_mask = (m_flat == 0)
        else:
            kv_pad_mask = None
            self_pad_mask = None

        # 포지셔널 임베딩
        Qi = self.pos(Qi)
        Kv = self.pos(Kv)

        # Fusion blocks
        x = Qi
        for blk in self.blocks:
            x = blk(x, Kv, kv_pad_mask=kv_pad_mask, self_pad_mask=self_pad_mask)

        # Global average over tokens → 128D + L2 normalize
        x = x.mean(dim=1)                # (B,d)
        x = self.head(x)                # (B,out_dim)
        x = F.normalize(x, dim=1)
        return x


# -------------------------------
# 5) 최종 모델: CombinedEmbedding (fusion 버전)
# -------------------------------
class CombinedEmbedding(nn.Module):
    """
    RGB(ResNet50) + Pattern(ResNet18, 1ch)
    → 1x1 proj로 공통 차원 d로 정렬
    → masked cross-attn → skip → self-attn → skip → FFN → skip
    → 128D 임베딩(L2 정규화)
    """
    def __init__(
        self,
        d: int = 256,            # 공통 임베딩 채널
        fused_dim: int = 128,    # 최종 출력 차원
        nblocks: int = 2,
        nheads: int = 4,
        pdrop: float = 0.1,
        pretrained_backbones: bool = True,
    ):
        super().__init__()
        self.rgb_backbone = ResNet50Feature(pretrained=pretrained_backbones)      # C=2048
        self.pat_backbone = PatternResNet18Feature(pretrained=pretrained_backbones)  # C=512

        # 해상도: 224기준 7x7. 다른 크기 입력도 자연스럽게 따라감.
        self.fusion = PatternFusion(
            c_img=self.rgb_backbone.out_channels,
            c_pat=self.pat_backbone.out_channels,
            d=d, H=7, W=7, nblocks=nblocks, nheads=nheads, pdrop=pdrop, out_dim=fused_dim
        )
        self.embed_dim = fused_dim

    def forward(self, img_rgb: torch.Tensor, bw_1ch: torch.Tensor) -> torch.Tensor:
        """
        img_rgb: (B,3,H,W)
        bw_1ch : (B,1,H,W)  (seg_green_bw: >0 전경)
        """
        # 1) 백본 feature map
        F_img = self.rgb_backbone(img_rgb)  # (B,2048,H/32,W/32)
        F_pat = self.pat_backbone(bw_1ch)   # (B, 512,H/32,W/32)

        # 2) foreground mask (bw>0) → fusion에 전달
        with torch.no_grad():
            fg_mask = (bw_1ch > 0).float()  # (B,1,H,W)

        # 3) fusion -> 128D
        emb = self.fusion(F_img, F_pat, fg_mask=fg_mask)  # (B,128), L2-normalized
        return emb