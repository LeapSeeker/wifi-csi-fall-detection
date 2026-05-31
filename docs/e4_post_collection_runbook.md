# E4 Post-Collection Runbook

E4 수집 완료 직후 빠르게 품질 확정과 다음 작업으로 넘어가기 위한 절차.

## 1. 정렬본 갱신

```bash
python tools/safesignal_debug.py clean-csv --env E4 --overwrite
```

모델 입력은 전처리에서 timestamp 정렬을 수행하므로, 수집 후 검토도 `data/cleaned` 기준으로 맞춘다.

## 2. 완료율 / 재수집 후보 확인

```bash
python tools/safesignal_debug.py post-collect --env 4 --dir data/cleaned --subject 2 3
```

파일로 남길 때:

```bash
python tools/safesignal_debug.py post-collect --env 4 --dir data/cleaned --subject 2 3 --out reports/e4_post_collect_latest.md
```

NO_MOTION까지 포함해서 최종 완료 확인할 때:

```bash
python tools/safesignal_debug.py post-collect --env 4 --dir data/cleaned --subject 2 3 --include-no-motion
```

## 3. 재수집 판단 기준

정렬 후 기준만 사용한다.

- `loss rate >= 10%`
- `absolute timestamp gap p95 >= 30ms`
- `absolute timestamp gap max >= 150ms`
- `pair_dt p95 >= 25ms`

`timestamp reversal`은 저장 순서 artifact라 재수집 기준으로 쓰지 않는다.

## 4. 수집 완료 후 모델 작업 순서

1. `post-collect`에서 missing trial이 없는지 확인한다.
2. Q3 기준 재수집 후보가 있으면 raw/cleaned에서 바로 삭제하지 말고 quarantine으로 이동한다.
3. `clean-csv --env E4 --overwrite`를 다시 실행한다.
4. SafeSignal CSV를 모델 cache로 변환하는 단계가 필요하다.
5. cache가 준비되면 `model/finetune/train.py`의 `--safesignal_cache`, `--alsaify_cache`를 지정해 fold별 학습을 진행한다.

현재 fine-tuning entrypoint는 CSV 직접 입력이 아니라 cache 입력을 요구한다.
