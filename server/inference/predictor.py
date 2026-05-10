"""FallPredictor — best.pt 로딩 + (300,104) 윈도우 → 클래스 확률.

child process 내부에서만 import (worker.py top-level import 금지).
"""
from __future__ import annotations

import sys
import warnings
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
from model.preprocessing.pipeline import window_to_model_input  # noqa: E402

from .config import FALL_CLASS_IDX, FALL_THRESHOLD, MODEL_PATH


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
        if ckpt_classes is not None:
            if len(ckpt_classes) != len(CLASSES) or tuple(ckpt_classes) != tuple(CLASSES):
                warnings.warn(
                    f"[FallPredictor] checkpoint classes {ckpt_classes} "
                    f"!= model CLASSES {CLASSES}. 7-class fine-tuned model? "
                    "현재 inference는 pretrained 6-class 기준입니다."
                )

        self.classes = tuple(ckpt_classes) if ckpt_classes else CLASSES

        self.model = CNNGRUAttention(n_classes=len(self.classes))
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, window: np.ndarray) -> dict:
        """(300, 104) → {class, confidence, is_fall, probabilities}."""
        if window.ndim != 2:
            raise ValueError(f"2D input required (n_t, n_sc). got {window.shape}")

        model_input = window_to_model_input(window)        # (1, 28, 20)
        x = torch.from_numpy(model_input).float()          # (1, 28, 20)
        x = x.unsqueeze(0).to(self.device)                 # (1, 1, 28, 20)

        logits = self.model(x)                             # (1, n_classes)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # (n_classes,)

        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        fall_conf = float(probs[FALL_CLASS_IDX])
        is_fall = (class_idx == FALL_CLASS_IDX) and (fall_conf >= FALL_THRESHOLD)

        return {
            "class": self.classes[class_idx],
            "confidence": confidence,
            "is_fall": is_fall,
            "probabilities": {self.classes[i]: float(probs[i]) for i in range(len(self.classes))},
        }
