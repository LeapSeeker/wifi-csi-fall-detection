"""사전학습 / 평가용 메트릭 모듈.

낙상 인식의 1차 지표는 fall 클래스 Recall (놓친 낙상 = 사용자 위험).
부가로 FAR(오탐률), F1, 전체 accuracy, 혼동 행렬을 함께 산출.

CLAUDE.md MVG 목표:
- fall Recall ≥ 0.85
- FAR        ≤ 0.15
- F1         ≥ 0.85

train.py 사용 패턴:
    from model.pretrained.metrics import compute_metrics, format_report

    val_loss, val_acc, y_true, y_pred = evaluate(...)
    m = compute_metrics(y_true, y_pred, classes=list(CLASSES))
    print(format_report(m))
    if m.fall_recall > best_recall: ...  # checkpoint
    save_metrics(m, ckpt_dir / "final_metrics.json")
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# ── 목표치 (CLAUDE.md MVG) ─────────────────────────────────────────────────
FALL_RECALL_TARGET: float = 0.85
FAR_TARGET: float = 0.15
F1_TARGET: float = 0.85

DEFAULT_FALL_LABEL: int = 0  # CLASSES[0] == "fall" 가정


# ── 데이터 클래스 ───────────────────────────────────────────────────────────

@dataclass
class FallMetrics:
    """fall 클래스 기준 메트릭 + 전체 accuracy + 혼동 행렬 + 목표 달성."""
    accuracy: float
    fall_recall: float
    fall_precision: float
    fall_f1: float
    far: float                         # FP / (FP + TN), fall 이진 시점
    tp: int
    fp: int
    fn: int
    tn: int
    confusion: list[list[int]]         # [n_classes][n_classes], row=true col=pred
    classes: list[str]
    counts: dict[str, int]             # 클래스별 true 샘플 수
    fall_label: int
    # 목표 달성 플래그
    meets_recall_target: bool
    meets_far_target: bool
    meets_f1_target: bool
    meets_all_targets: bool
    # 사용한 임계치 (재현성 위해 함께 저장)
    targets: dict[str, float] = field(default_factory=dict)


# ── 핵심 계산 ───────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    classes: Sequence[str],
    fall_label: int = DEFAULT_FALL_LABEL,
    recall_target: float = FALL_RECALL_TARGET,
    far_target: float = FAR_TARGET,
    f1_target: float = F1_TARGET,
) -> FallMetrics:
    """fall 클래스 기준 메트릭 + 전체 accuracy + 혼동 행렬을 계산.

    Parameters
    ----------
    y_true, y_pred : 1D int 시퀀스
        같은 길이의 정수 라벨.
    classes : 길이 n_classes 문자열 시퀀스
        혼동 행렬과 카운트에 표시될 이름.
    fall_label : int
        fall 클래스 인덱스. 기본 0 (CLASSES[0]=='fall').
    recall_target, far_target, f1_target : float
        MVG 임계. CLAUDE.md 기본 (0.85, 0.15, 0.85).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape} y_pred={y_pred.shape}")
    n = len(y_true)
    n_classes = len(classes)

    # 혼동 행렬 (벡터화)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    if n > 0:
        np.add.at(cm, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)

    # 전체 accuracy
    acc = float((y_true == y_pred).mean()) if n else 0.0

    # fall 이진 시점
    pred_pos = (y_pred == fall_label)
    true_pos = (y_true == fall_label)
    tp = int((pred_pos & true_pos).sum())
    fp = int((pred_pos & ~true_pos).sum())
    fn = int((~pred_pos & true_pos).sum())
    tn = int((~pred_pos & ~true_pos).sum())

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * recall * precision / (recall + precision)
          if (recall + precision) else 0.0)
    far = fp / (fp + tn) if (fp + tn) else 0.0

    counts = {classes[i]: int((y_true == i).sum()) for i in range(n_classes)}

    meets_recall = recall >= recall_target
    meets_far = far <= far_target
    meets_f1 = f1 >= f1_target

    return FallMetrics(
        accuracy=acc,
        fall_recall=recall,
        fall_precision=precision,
        fall_f1=f1,
        far=far,
        tp=tp, fp=fp, fn=fn, tn=tn,
        confusion=cm.tolist(),
        classes=list(classes),
        counts=counts,
        fall_label=fall_label,
        meets_recall_target=meets_recall,
        meets_far_target=meets_far,
        meets_f1_target=meets_f1,
        meets_all_targets=meets_recall and meets_far and meets_f1,
        targets={
            "fall_recall_target": recall_target,
            "far_target": far_target,
            "f1_target": f1_target,
        },
    )


# ── 출력 / 직렬화 ──────────────────────────────────────────────────────────

def _flag(ok: bool) -> str:
    return "✓" if ok else "✗"


def format_report(m: FallMetrics, header: str | None = None) -> str:
    """사람이 읽기 좋은 텍스트 리포트 생성."""
    cls = m.classes
    cm = m.confusion
    fall_name = cls[m.fall_label]
    name_w = max(len(c) for c in cls) + 1

    lines: list[str] = []
    if header:
        lines.append(header)
    lines.append("=" * 64)
    lines.append(f"전체 Accuracy : {m.accuracy:.4f}  ({sum(m.counts.values())} samples)")
    lines.append(f"Fall = '{fall_name}' (label={m.fall_label})")
    lines.append(
        f"  TP={m.tp:>4} FP={m.fp:>4} FN={m.fn:>4} TN={m.tn:>4}"
    )
    lines.append(
        f"  Recall    : {m.fall_recall:.4f}  {_flag(m.meets_recall_target)}  "
        f"(target ≥ {m.targets['fall_recall_target']:.2f})  ← MVG"
    )
    lines.append(
        f"  Precision : {m.fall_precision:.4f}"
    )
    lines.append(
        f"  F1        : {m.fall_f1:.4f}  {_flag(m.meets_f1_target)}  "
        f"(target ≥ {m.targets['f1_target']:.2f})"
    )
    lines.append(
        f"  FAR       : {m.far:.4f}  {_flag(m.meets_far_target)}  "
        f"(target ≤ {m.targets['far_target']:.2f})  (False Alarm Rate)"
    )
    lines.append(
        f"  ALL TARGETS: {_flag(m.meets_all_targets)}"
    )

    # 혼동 행렬
    lines.append("")
    lines.append("Confusion Matrix (row=true, col=pred):")
    col_w = max(8, max(len(c) for c in cls))
    header_row = " " * (name_w + 3) + " ".join(f"{c:>{col_w}}" for c in cls)
    lines.append(header_row)
    for i, row in enumerate(cm):
        marker = " ←fall" if i == m.fall_label else ""
        row_str = " ".join(f"{v:>{col_w}d}" for v in row)
        lines.append(f"  {cls[i]:<{name_w}} | {row_str}{marker}")

    # 클래스별 카운트
    lines.append("")
    lines.append("True 샘플 수: " + ", ".join(f"{k}={v}" for k, v in m.counts.items()))
    return "\n".join(lines)


def save_metrics(m: FallMetrics, path: str | Path) -> None:
    """JSON으로 저장 (dataclass → dict)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(m), indent=2, ensure_ascii=False))


def load_metrics(path: str | Path) -> FallMetrics:
    """JSON에서 로드. 누락 필드는 None — 직렬화 호환용."""
    data = json.loads(Path(path).read_text())
    return FallMetrics(**data)


# ── 시각화 ─────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
    save_path: str | Path | None = None,
    fall_label: int = DEFAULT_FALL_LABEL,
) -> None:
    """행 정규화 혼동 행렬을 그리고 fall 행/열을 강조 표시.

    각 셀에는 "비율%\\n(원시 카운트)" 를 함께 주석. 제목에는 기존 FallMetrics
    에서 뽑은 Recall / FAR 값을 표시. ``save_path`` 가 주어지면 PNG 로 저장,
    아니면 ``plt.show()`` 호출.

    matplotlib·sklearn 임포트는 함수 내부에서만 수행하여 학습 경로(train.py)의
    런타임 의존성을 늘리지 않는다.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n_classes = len(class_names)
    labels = list(range(n_classes))

    # 원시 카운트와 행 정규화 (true 기준 — recall 해석)
    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_raw, row_sums,
        where=row_sums != 0,
        out=np.zeros(cm_raw.shape, dtype=float),
    )

    # 기존 FallMetrics 에서 Recall / FAR 재사용 (제목에 표시)
    m = compute_metrics(y_true, y_pred, classes=class_names, fall_label=fall_label)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized rate")

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(n_classes):
        for j in range(n_classes):
            pct = cm_norm[i, j] * 100
            cnt = int(cm_raw[i, j])
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j, i, f"{pct:.1f}%\n({cnt})",
                ha="center", va="center",
                color=text_color, fontsize=9,
            )

    # fall 행/열 강조 — 빨간 테두리 + 틱 라벨 색/굵기
    fall_color = "crimson"
    ax.add_patch(Rectangle(
        (-0.5, fall_label - 0.5), n_classes, 1,
        fill=False, edgecolor=fall_color, lw=2.5,
    ))
    ax.add_patch(Rectangle(
        (fall_label - 0.5, -0.5), 1, n_classes,
        fill=False, edgecolor=fall_color, lw=2.5,
    ))
    ax.get_xticklabels()[fall_label].set_color(fall_color)
    ax.get_xticklabels()[fall_label].set_fontweight("bold")
    ax.get_yticklabels()[fall_label].set_color(fall_color)
    ax.get_yticklabels()[fall_label].set_fontweight("bold")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fall_name = class_names[fall_label]
    ax.set_title(
        f"Confusion Matrix (row-normalized)\n"
        f"Fall='{fall_name}'  Recall={m.fall_recall:.3f}  FAR={m.far:.3f}"
    )

    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ── 단위 테스트 ────────────────────────────────────────────────────────────

def _self_test() -> None:
    classes = ["fall", "walking", "sit_stand", "lying", "standing", "picking"]

    # ── 케이스 1: 완벽 ─────────────────────────────────────────────────
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    m = compute_metrics(y, y, classes)
    assert m.accuracy == 1.0
    assert m.fall_recall == 1.0
    assert m.far == 0.0
    assert m.meets_all_targets
    print("[1] 완벽 예측 → accuracy=1, recall=1, FAR=0, all targets met ✓")

    # ── 케이스 2: fall 절반 놓침 ─────────────────────────────────────
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    m = compute_metrics(y_true, y_pred, classes)
    assert m.tp == 2 and m.fn == 2
    assert m.fall_recall == 0.5  # 4 fall 중 2개만 잡음
    assert not m.meets_recall_target  # 0.5 < 0.85
    assert m.far == 0.0  # 비-fall에서 fall로 오예측 없음
    assert m.meets_far_target
    print(f"[2] 절반 fall 놓침 → recall={m.fall_recall:.2f} ✗, FAR={m.far:.2f} ✓")

    # ── 케이스 3: 과도한 오탐 ──────────────────────────────────────────
    y_true = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 0, 0, 1, 0, 0, 2, 2])
    m = compute_metrics(y_true, y_pred, classes)
    # tp=2, fp=5(non-fall→fall로 5개), fn=0, tn=3
    assert m.tp == 2 and m.fp == 5 and m.fn == 0 and m.tn == 3
    assert m.fall_recall == 1.0  # 모든 실제 fall은 잡음
    assert m.far == 5/8  # 5/(5+3) = 0.625
    assert m.meets_recall_target
    assert not m.meets_far_target  # FAR 너무 높음
    assert not m.meets_all_targets
    print(f"[3] 과도 오탐 → recall={m.fall_recall:.2f} ✓, FAR={m.far:.3f} ✗ ALL=✗")

    # ── 케이스 4: 혼동 행렬 정합 ─────────────────────────────────────
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2])
    m = compute_metrics(y_true, y_pred, classes)
    cm = np.array(m.confusion)
    assert cm.shape == (6, 6)
    assert cm[0, 0] == 1  # fall→fall
    assert cm[0, 1] == 2  # fall→walking
    assert cm[1, 1] == 1  # walking→walking
    assert cm[1, 0] == 1  # walking→fall
    assert cm[2, 2] == 1  # sit_stand→sit_stand
    assert cm.sum() == len(y_true)
    print(f"[4] 혼동 행렬 합={cm.sum()} (=y_true 길이 {len(y_true)}) 및 셀 정합 ✓")

    # ── 케이스 5: JSON round-trip ────────────────────────────────────
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp = Path(f.name)
    save_metrics(m, tmp)
    loaded = load_metrics(tmp)
    assert loaded.accuracy == m.accuracy
    assert loaded.confusion == m.confusion
    assert loaded.targets == m.targets
    tmp.unlink()
    print("[5] JSON 직렬화 round-trip 정확 ✓")

    # ── 케이스 6: 빈 입력 ─────────────────────────────────────────────
    m_empty = compute_metrics([], [], classes)
    assert m_empty.accuracy == 0.0
    assert m_empty.tp == 0
    assert sum(m_empty.counts.values()) == 0
    print("[6] 빈 입력 → 0으로 안전 처리 ✓")

    # ── 리포트 출력 (수동 점검용) ───────────────────────────────────
    print("\n" + "=" * 64)
    print("샘플 리포트 (케이스 3 — 과도 오탐):")
    print(format_report(
        compute_metrics(
            np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
            np.array([0, 0, 0, 0, 0, 1, 0, 0, 2, 2]),
            classes,
        ),
        header="[Validation Report]",
    ))

    print("\nPASS — 6개 검증 모두 통과")


def _demo_plot() -> None:
    """7-클래스 더미 라벨로 시각화 데모. `python metrics.py` 실행 시 함께 표시.

    참고: 사전학습 파이프라인은 6 클래스 (Alsaify 매핑상 'running' 미존재)지만,
    파인튜닝 단계에서는 'running' 이 추가된 7 클래스 구성이므로 데모도 7 클래스
    로 둔다. (model/CLAUDE.md 「달리기(빠른 보행): 파인튜닝 전용」)
    """
    rng = np.random.default_rng(0)
    classes = ["fall", "walking", "sit_stand", "lying",
               "standing", "running", "picking"]
    n_per_class = [60, 80, 70, 60, 80, 50, 60]

    # 클래스별 정확도(대각) ≈ correct_rates, 나머지는 다른 클래스로 분산.
    # 비-fall 의 일부 오답은 fall 로 보내 FAR 가 0이 아닌 시나리오를 생성.
    correct_rates = [0.83, 0.92, 0.88, 0.90, 0.85, 0.80, 0.88]
    y_true_parts, y_pred_parts = [], []
    for cls_idx, n in enumerate(n_per_class):
        y_true_parts.append(np.full(n, cls_idx, dtype=np.int64))
        n_correct = int(round(n * correct_rates[cls_idx]))
        n_wrong = n - n_correct
        wrong_choices = [c for c in range(len(classes)) if c != cls_idx]
        wrong = rng.choice(wrong_choices, size=n_wrong).astype(np.int64)
        if cls_idx != 0 and n_wrong > 0:
            n_to_fall = max(1, n_wrong // 3)
            wrong[:n_to_fall] = 0
        y_pred_parts.append(np.concatenate([
            np.full(n_correct, cls_idx, dtype=np.int64), wrong,
        ]))

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)

    m = compute_metrics(y_true, y_pred, classes=classes)
    print("\n" + "=" * 64)
    print("Demo metrics (7-class dummy, fall=index 0):")
    print(f"  Accuracy  : {m.accuracy:.4f}  ({sum(m.counts.values())} samples)")
    print(f"  Recall    : {m.fall_recall:.4f}   (fall TP/(TP+FN))")
    print(f"  Precision : {m.fall_precision:.4f}")
    print(f"  F1        : {m.fall_f1:.4f}")
    print(f"  FAR       : {m.far:.4f}        (fall FP/(FP+TN))")
    print("\nplot_confusion_matrix 호출 — 창을 닫으면 종료됩니다.")
    plot_confusion_matrix(y_true, y_pred, class_names=classes)


if __name__ == "__main__":
    try:
        _self_test()
        _demo_plot()
    except KeyboardInterrupt:
        print("\n중단됨.")
