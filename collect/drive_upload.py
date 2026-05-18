"""Optional Google Drive upload for collected SafeSignal CSV files.

This module intentionally depends on an external ``rclone`` installation instead
of embedding Google OAuth/API credentials in the project. Local CSV save remains
the source of truth; upload is a best-effort post-save step.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import threading
from datetime import datetime


ENV_UPLOAD = "SAFESIGNAL_DRIVE_UPLOAD"
ENV_REMOTE = "SAFESIGNAL_DRIVE_REMOTE"
ENV_RCLONE = "SAFESIGNAL_RCLONE_BIN"
ENV_TIMEOUT = "SAFESIGNAL_DRIVE_TIMEOUT_SEC"

DEFAULT_RCLONE = "rclone"
DEFAULT_TIMEOUT_SEC = 120
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_LOG = PROJECT_ROOT / "data" / "upload_log.md"
SAFESIGNAL_CSV_RE = re.compile(
    r"^E(?P<env>\d+)_S(?P<subject>\d+)_A_(?P<activity>.+)_T(?P<trial>\d+)\.csv$",
    re.IGNORECASE,
)

_LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class DriveUploadConfig:
    enabled: bool
    remote: str | None
    rclone_bin: str = DEFAULT_RCLONE
    timeout_sec: int = DEFAULT_TIMEOUT_SEC

    @classmethod
    def from_env(cls) -> "DriveUploadConfig":
        enabled_raw = os.environ.get(ENV_UPLOAD, "").strip().lower()
        enabled = enabled_raw in ("1", "true", "yes", "y", "on")

        timeout_raw = os.environ.get(ENV_TIMEOUT, "").strip()
        timeout_sec = DEFAULT_TIMEOUT_SEC
        if timeout_raw:
            try:
                timeout_sec = max(1, int(timeout_raw))
            except ValueError:
                timeout_sec = DEFAULT_TIMEOUT_SEC

        return cls(
            enabled=enabled,
            remote=os.environ.get(ENV_REMOTE),
            rclone_bin=os.environ.get(ENV_RCLONE, DEFAULT_RCLONE),
            timeout_sec=timeout_sec,
        )


@dataclass(frozen=True)
class UploadResult:
    attempted: bool
    success: bool
    message: str
    remote_path: str | None = None


def _remote_file_path(remote_root: str, local_path: Path) -> str:
    remote_root = remote_root.rstrip("/\\")
    m = SAFESIGNAL_CSV_RE.match(local_path.name)
    if not m:
        return f"{remote_root}/{local_path.name}"

    env_dir = f"E{int(m['env'])}"
    subject_dir = f"S{int(m['subject']):02d}"
    return f"{remote_root}/{env_dir}/{subject_dir}/{local_path.name}"


def _append_upload_log(local_path: Path, result: UploadResult) -> None:
    UPLOAD_LOG.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "OK" if result.success else "FAIL"
    remote = result.remote_path or "-"
    line = (
        f"| {now} | {status} | {local_path} | {remote} | "
        f"{result.message.replace('|', '/') } |\n"
    )

    with _LOG_LOCK:
        if not UPLOAD_LOG.exists():
            UPLOAD_LOG.write_text(
                "| time | status | local_path | remote_path | message |\n"
                "|---|---|---|---|---|\n",
                encoding="utf-8",
            )
        with UPLOAD_LOG.open("a", encoding="utf-8") as f:
            f.write(line)


def upload_file(local_path: str | Path, config: DriveUploadConfig | None = None) -> UploadResult:
    """Upload one CSV file with rclone.

    Returns an UploadResult and appends a small log entry when upload is enabled.
    """
    cfg = config or DriveUploadConfig.from_env()
    path = Path(local_path)

    if not cfg.enabled:
        return UploadResult(False, False, "upload disabled")

    if not cfg.remote:
        result = UploadResult(
            True,
            False,
            f"{ENV_REMOTE} is not set",
        )
        _append_upload_log(path, result)
        return result

    if not path.exists():
        result = UploadResult(True, False, f"local file not found: {path}")
        _append_upload_log(path, result)
        return result

    remote_path = _remote_file_path(cfg.remote, path)
    cmd = [
        cfg.rclone_bin,
        "copyto",
        str(path),
        remote_path,
    ]

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_sec,
        )
    except FileNotFoundError:
        result = UploadResult(
            True,
            False,
            f"rclone not found: {cfg.rclone_bin}",
            remote_path=remote_path,
        )
        _append_upload_log(path, result)
        return result
    except subprocess.TimeoutExpired:
        result = UploadResult(
            True,
            False,
            f"rclone timeout after {cfg.timeout_sec}s",
            remote_path=remote_path,
        )
        _append_upload_log(path, result)
        return result

    if completed.returncode == 0:
        result = UploadResult(True, True, "uploaded", remote_path=remote_path)
    else:
        stderr = (completed.stderr or completed.stdout or "").strip()
        result = UploadResult(
            True,
            False,
            stderr[:500] if stderr else f"rclone exited with {completed.returncode}",
            remote_path=remote_path,
        )
    _append_upload_log(path, result)
    return result


def upload_file_async(
    local_path: str | Path,
    config: DriveUploadConfig | None = None,
) -> threading.Thread | None:
    """Start a background upload thread when enabled.

    The thread is non-daemon so Python will not silently drop an in-flight upload
    when the user exits immediately after saving a session.
    """
    cfg = config or DriveUploadConfig.from_env()
    if not cfg.enabled:
        return None

    path = Path(local_path)

    def _worker() -> None:
        print(f"  [Drive] 업로드 시작: {path.name}")
        result = upload_file(path, cfg)
        if result.success:
            print(f"  [Drive] 업로드 완료: {result.remote_path}")
        else:
            print(f"  [Drive] 업로드 실패: {result.message}")

    thread = threading.Thread(target=_worker, name=f"drive-upload-{path.name}")
    thread.start()
    return thread


def upload_status_message(config: DriveUploadConfig | None = None) -> str:
    cfg = config or DriveUploadConfig.from_env()
    if not cfg.enabled:
        return "[Drive] 자동 업로드 비활성화 (SAFESIGNAL_DRIVE_UPLOAD=1 로 활성화)"
    if not cfg.remote:
        return f"[Drive] 자동 업로드 활성화 실패: {ENV_REMOTE} 미설정"
    return f"[Drive] 자동 업로드 활성화: {cfg.remote}"
