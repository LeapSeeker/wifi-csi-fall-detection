"""InferenceWorker — 별도 process에서 추론 실행 (D-015).

- top-level은 가벼움: multiprocessing/queue만 import.
- predictor / buffer 는 child process 내부(_inference_process)에서 import.
- main process가 torch/RPCA를 안 불러야 self-check가 가벼움.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import sys
import traceback
from pathlib import Path
from typing import Optional

from .config import MODEL_PATH


def _inference_process(
    input_queue: "mp.Queue",
    result_queue: "mp.Queue",
    model_path: str,
):
    """child process 진입점. heavy import 여기서 수행."""
    project_root = Path(__file__).resolve().parents[2]
    server_dir = project_root / "server"
    for p in (str(project_root), str(server_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from inference.predictor import FallPredictor  # type: ignore
        from inference.buffer import SlidingWindowBuffer  # type: ignore
    except ImportError:
        # relative-import 안전망
        from server.inference.predictor import FallPredictor  # type: ignore
        from server.inference.buffer import SlidingWindowBuffer  # type: ignore

    try:
        predictor = FallPredictor(model_path=model_path)
    except Exception:
        traceback.print_exc()
        result_queue.put({"error": "FallPredictor init failed"})
        return

    buffer = SlidingWindowBuffer()
    print(f"[InferenceWorker] started. model={model_path}")

    while True:
        try:
            rx1, rx2 = input_queue.get()
        except (EOFError, KeyboardInterrupt):
            return
        except Exception:
            traceback.print_exc()
            continue

        try:
            triggered = buffer.add(
                rx1["amplitudes"],
                rx2["amplitudes"],
                timestamp_us=int(rx1.get("timestamp_us", 0)),
            )
            if not triggered:
                continue

            window = buffer.get_window()
            result = predictor.predict(window)
            result["seq_num"] = int(rx1.get("seq_num", 0))
            result["timestamp_us"] = int(
                buffer.window_timestamp_us or rx1.get("timestamp_us", 0)
            )
            result_queue.put(result)
        except Exception:
            traceback.print_exc()
            try:
                result_queue.put({"error": "predict failed"})
            except Exception:
                pass


class InferenceWorker:
    """input_queue → child process(predict) → result_queue.

    main process에서는 가벼운 큐만 보유. start() 호출 전에는 프로세스 없음.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = str(model_path)
        ctx = mp.get_context("spawn")  # Windows 안전
        self.input_queue: "mp.Queue" = ctx.Queue(maxsize=500)
        self.result_queue: "mp.Queue" = ctx.Queue()
        self._ctx = ctx
        self._process: Optional[mp.Process] = None
        self._drop_count = 0

    def start(self):
        if self._process is not None and self._process.is_alive():
            return
        self._process = self._ctx.Process(
            target=_inference_process,
            args=(self.input_queue, self.result_queue, self.model_path),
            daemon=True,
        )
        self._process.start()

    def put(self, rx1: dict, rx2: dict):
        try:
            self.input_queue.put_nowait((rx1, rx2))
        except queue.Full:
            self._drop_count += 1
            if self._drop_count % 100 == 1:
                print(f"[InferenceWorker] input_queue full - drop #{self._drop_count}")

    def get_result(self) -> Optional[dict]:
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
