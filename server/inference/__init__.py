"""server.inference — 추론 파이프라인 모듈 (D-013/D-014/D-015).

InferenceWorker만 외부에 노출. predictor/buffer는 child process 내부에서
import하여 main process가 torch/model heavy import를 부담하지 않게 한다.
"""
from .worker import InferenceWorker

__all__ = ["InferenceWorker"]
