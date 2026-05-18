"""Pi4 WebSocket message contract.

This module is the server-side source of truth for JSON payloads sent to Pi4.
"""
from __future__ import annotations

PI4_EVENT_FALL_DETECTED = "fall_detected"
PI4_LABEL_FALL = "fall"


def build_fall_alert_payload(
    confidence: float = 0.0,
    seq_num: int = 0,
    timestamp_us: int = 0,
) -> dict:
    """Build the legacy Pi4 fall alert payload."""
    return {
        "event": PI4_EVENT_FALL_DETECTED,
        "label": PI4_LABEL_FALL,
        "confidence": float(confidence),
        "seq_num": int(seq_num),
        "timestamp_us": int(timestamp_us),
    }
