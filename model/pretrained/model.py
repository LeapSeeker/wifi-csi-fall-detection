"""CNN + GRU + Temporal Attention 사전학습 모델.

입력: (N, 1, 28, 20)  ← preprocessing.window_to_model_input 출력
출력: (N, 6) logits

shape 흐름
----------
(N, 1,  28, 20)            입력 (channel=1, time=28, lag=20)
(N, 16, 28, 10)  Conv→Pool(lag/2)
(N, 32, 28,  5)  Conv→Pool(lag/2)
(N, 64, 28,  5)  Conv
(N, 28, 320)     permute+flatten  → GRU 입력 (T=28, F=64*5)
(N, 28, 256)     BiGRU(hidden=128)
(N, 256)         Temporal Attention (T 축 가중합)
(N, 6)           FC

설계 메모
---------
- 시간 축(28)은 CNN 단계에서 보존 → GRU가 28 step 시계열로 처리.
  (kernel time=3 + padding=1, MaxPool은 lag 축에만 적용 (1,2)).
- GRU는 bidirectional 기본. 짧은 시퀀스(28)에서 양방향 정보가 attention과 결합 시 유리.
- Attention은 Bahdanau-style additive: e_t = v^T tanh(W h_t).
  최종 클래스 결정에 어느 sub-window가 기여했는지 시각화 가능 (return_attention=True).
- 6 클래스 (사전학습 기준, 달리기 제외) — model/CLAUDE.md 표 준수.
- 추론 출력은 최종적으로 fall/walking/sit_stand/lying/standing/picking로 매핑
  (SHARED.md 라벨 규약). 'running'은 파인튜닝 단계에서 추가.

참고
----
XFall (IEEE JSAC 2024): Attention + SDP 조합 검증.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 사전학습 클래스 (model/CLAUDE.md 매핑 표 기준, SHARED.md 라벨명과 정합)
CLASSES: tuple[str, ...] = (
    "fall",      # Alsaify A2, A5
    "walking",   # A6, A8
    "sit_stand", # A10, A11
    "lying",     # A3 (C1+C2)
    "standing",  # A4
    "picking",   # A12
)
N_CLASSES = len(CLASSES)


class CNNFeatureExtractor(nn.Module):
    """(N, 1, 28, 20) → (N, T=28, F).

    시간 축 보존, lag 축만 다운샘플 후 채널과 결합해 GRU 입력 차원 F = c3*5 생성.
    """

    def __init__(
        self,
        conv_channels: tuple[int, int, int] = (16, 32, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        c1, c2, c3 = conv_channels
        self.block1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # lag 20 → 10
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # lag 10 → 5
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_features = c3 * 5  # GRU input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)            # (N, c1, 28, 10)
        x = self.block2(x)            # (N, c2, 28, 5)
        x = self.block3(x)            # (N, c3, 28, 5)
        x = self.dropout(x)
        x = x.permute(0, 2, 1, 3)     # (N, 28, c3, 5)
        x = x.flatten(2)              # (N, 28, c3*5)
        return x


class TemporalAttention(nn.Module):
    """Bahdanau-style additive attention over T 축.

    h: (N, T, H) → context: (N, H), weights: (N, T)
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 64):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = torch.tanh(self.W(h))                  # (N, T, A)
        scores = self.v(e).squeeze(-1)             # (N, T)
        weights = F.softmax(scores, dim=1)         # (N, T)
        context = (h * weights.unsqueeze(-1)).sum(dim=1)  # (N, H)
        return context, weights


class CNNGRUAttention(nn.Module):
    """Pretraining 모델: CNN → BiGRU → Temporal Attention → FC."""

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        conv_channels: tuple[int, int, int] = (16, 32, 64),
        gru_hidden: int = 128,
        gru_layers: int = 1,
        bidirectional: bool = True,
        attn_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cnn = CNNFeatureExtractor(conv_channels=conv_channels, dropout=dropout)
        self.gru = nn.GRU(
            input_size=self.cnn.out_features,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        gru_out_dim = gru_hidden * (2 if bidirectional else 1)
        self.attention = TemporalAttention(gru_out_dim, attn_dim=attn_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"입력 shape (N,1,T,F) 필요. got {tuple(x.shape)}")
        feat = self.cnn(x)                  # (N, 28, F)
        h, _ = self.gru(feat)               # (N, 28, gru_out_dim)
        context, weights = self.attention(h)
        logits = self.classifier(context)   # (N, n_classes)
        if return_attention:
            return logits, weights
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = CNNGRUAttention()
    model.eval()

    x = torch.randn(4, 1, 28, 20)
    with torch.no_grad():
        logits = model(x)
        logits2, attn = model(x, return_attention=True)

    print(f"input  : {tuple(x.shape)}")
    print(f"logits : {tuple(logits.shape)}  (expected (4, {N_CLASSES}))")
    print(f"attn   : {tuple(attn.shape)}    softmax sum per sample = "
          f"{attn.sum(dim=1).tolist()}")
    print(f"params : {count_parameters(model):,}")
    print(f"classes: {CLASSES}")
