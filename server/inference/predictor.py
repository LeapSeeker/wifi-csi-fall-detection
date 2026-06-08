"""FallPredictor — best.pt 로딩 + (300,104) 윈도우 → 클래스 확률.

child process 내부에서만 import (worker.py top-level import 금지).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import numpy as np

# project_root 경로 추가 → model.* import
# parents[0]=inference, parents[1]=server, parents[2]=project_root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model.pretrained.model import CLASSES, CNNGRUAttention  # noqa: E402

from .energy import ENERGY_METRICS, compute_window_energy
from .config import (
    ENERGY_GATE_ENABLED,
    ENERGY_GATE_METRIC,
    ENERGY_GATE_THRESHOLD,
    FALL_CONSECUTIVE_N,
    FALL_LABELS,
    FALL_THRESHOLD,
    MODEL_PATH,
)


def _model_input_and_energy(
    window: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """window_to_model_input()과 동일 경로로 z-score 전 SDP energy를 함께 계산.

    energy 계산은 calibrate 도구와 공유하는 단일 helper(compute_window_energy)에
    위임한다(RPCA 1회). 반환된 SDP를 여기서 z-score 하여 모델 입력으로 만든다.
    """
    sdp, metrics = compute_window_energy(window)
    sdp_z = (sdp - sdp.mean()) / (sdp.std() + 1e-6)
    return sdp_z[None, ...], metrics


class FallPredictor:
    """best.pt 로드 → predict(window: (300,104)) → dict."""

    def __init__(
        self,
        model_path: Union[str, Path] = MODEL_PATH,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"best.pt not found: {model_path}")

        ckpt = torch.load(model_path, map_location=self.device)
        ckpt_classes = ckpt.get("classes")
        if ckpt_classes is None:
            raise ValueError(
                "[FallPredictor] checkpoint has no 'classes'. "
                f"Expected pretrained6 classes {CLASSES}."
            )
        if tuple(ckpt_classes) != tuple(CLASSES):
            raise ValueError(
                f"[FallPredictor] checkpoint classes {tuple(ckpt_classes)} "
                f"!= expected pretrained6 classes {CLASSES}."
            )

        self.classes = tuple(ckpt_classes)
        self.fall_class_indices = tuple(
            i for i, name in enumerate(self.classes) if name in FALL_LABELS
        )
        if self.fall_class_indices != (0,):
            raise ValueError(
                f"[FallPredictor] fall_class_indices={self.fall_class_indices}, "
                "expected (0,) for pretrained6 inference."
            )

        self.model = CNNGRUAttention(n_classes=len(self.classes))
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()
        self.model.to(self.device)
        self.threshold = float(FALL_THRESHOLD)
        self.consecutive_n = int(FALL_CONSECUTIVE_N)
        self.energy_gate_enabled = bool(ENERGY_GATE_ENABLED)
        self.energy_gate_metric = str(ENERGY_GATE_METRIC)
        self.energy_gate_threshold = float(ENERGY_GATE_THRESHOLD)
        self._consecutive_fire = 0
        self.energy_gate_skipped = 0
        self.energy_gate_passed = 0

        if self.energy_gate_metric not in ENERGY_METRICS:
            raise ValueError(
                f"Unknown ENERGY_GATE_METRIC={self.energy_gate_metric!r}. "
                f"valid={sorted(ENERGY_METRICS)}"
            )

    @torch.no_grad()
    def predict(self, window: np.ndarray) -> dict:
        """(300, 104) → {class, confidence, is_fall, probabilities}."""
        if window.ndim != 2:
            raise ValueError(f"2D input required (n_t, n_sc). got {window.shape}")

        model_input, energy = _model_input_and_energy(window)  # (1, 28, 20)
        gate_value = float(energy[self.energy_gate_metric])
        if self.energy_gate_enabled and gate_value < self.energy_gate_threshold:
            self.energy_gate_skipped += 1
            self._consecutive_fire = 0
            return {
                "class": "non_fall_energy_gate",
                "confidence": 0.0,
                "is_fall": False,
                "raw_is_fall": False,
                "fall_confidence": 0.0,
                "probabilities": {name: 0.0 for name in self.classes},
                "energy_gate": {
                    "enabled": True,
                    "skipped": True,
                    "metric": self.energy_gate_metric,
                    "value": gate_value,
                    "threshold": self.energy_gate_threshold,
                    "skipped_count": self.energy_gate_skipped,
                    "passed_count": self.energy_gate_passed,
                    "metrics": energy,
                },
                "fall_consecutive": {
                    "n": self.consecutive_n,
                    "count": self._consecutive_fire,
                },
            }
        self.energy_gate_passed += 1

        x = torch.from_numpy(model_input).float()          # (1, 28, 20)
        x = x.unsqueeze(0).to(self.device)                 # (1, 1, 28, 20)

        logits = self.model(x)                             # (1, n_classes)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # (n_classes,)

        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        class_name = self.classes[class_idx]
        fall_conf = (
            float(max(probs[i] for i in self.fall_class_indices))
            if self.fall_class_indices else 0.0
        )
        # 0순위 판정식 정렬: 학습 평가 predict_with_fall_threshold()와 동일하게
        # argmax class와 무관하게 fall_confidence >= threshold 면 raw fall로 본다.
        # (기존 argmax==fall AND 조건은 평가식과 불일치 → 제거. 정적 오발은
        #  energy gate / N consecutive 로 막는다.)
        raw_is_fall = fall_conf >= self.threshold
        if raw_is_fall:
            self._consecutive_fire += 1
        else:
            self._consecutive_fire = 0
        is_fall = raw_is_fall and self._consecutive_fire >= self.consecutive_n

        return {
            "class": class_name,
            "confidence": confidence,
            "is_fall": is_fall,
            "raw_is_fall": raw_is_fall,
            "fall_confidence": fall_conf,
            "probabilities": {self.classes[i]: float(probs[i]) for i in range(len(self.classes))},
            "threshold": self.threshold,
            "energy_gate": {
                "enabled": self.energy_gate_enabled,
                "skipped": False,
                "metric": self.energy_gate_metric,
                "value": gate_value,
                "threshold": self.energy_gate_threshold,
                "skipped_count": self.energy_gate_skipped,
                "passed_count": self.energy_gate_passed,
                "metrics": energy,
            },
            "fall_consecutive": {
                "n": self.consecutive_n,
                "count": self._consecutive_fire,
            },
        }
