# server/run_server.ps1
# 데모 최종 운영값으로 추론 서버를 띄운다.
#
# 왜 .env가 아니라 이 스크립트인가:
#   config.py 의 MODEL_PATH / FALL_THRESHOLD 는 import 시점(= main.py 맨 위
#   inference.worker import 시)에 os.getenv 로 읽힌다. load_dotenv() 는 그
#   뒤 main() 안에서야 호출되므로, 모델 경로는 .env 로는 안 먹고 반드시
#   "프로세스 시작 전 실제 환경변수" 여야 한다.
#
# 사용:  cd server ; .\run_server.ps1
#
# 데모 주의:
#   RPi4는 2.4GHz WiFi에 붙이면 ESP32 CSI를 교란해 정지 오발이 크게 증가한다.
#   데모 전 RPi4를 유선 Ethernet / 5GHz 전용 / WiFi off 중 하나로 고정한다.

$env:SAFESIGNAL_MODEL_PATH = "model/finetune/checkpoints_item4_balanced/onset_primary_balanced_aug_s42/best_operating.pt"
$env:SAFESIGNAL_FALL_THRESHOLD = "0.30"
$env:SAFESIGNAL_INFERENCE_STRIDE = "150"
$env:SAFESIGNAL_RPCA_MAX_ITER = "100"
$env:SAFESIGNAL_FALL_CONSECUTIVE_N = "1"
$env:SAFESIGNAL_ENERGY_GATE_ENABLED = "0"   # gate off (낙상/정지 분리 실패)
$env:SAFESIGNAL_CONF_DIAG = "1"
$env:COOLDOWN_SEC = "10"

Write-Host "[run_server] MODEL_PATH        = $env:SAFESIGNAL_MODEL_PATH"
Write-Host "[run_server] FALL_THRESHOLD    = $env:SAFESIGNAL_FALL_THRESHOLD"
Write-Host "[run_server] INFERENCE_STRIDE  = $env:SAFESIGNAL_INFERENCE_STRIDE"
Write-Host "[run_server] RPCA_MAX_ITER     = $env:SAFESIGNAL_RPCA_MAX_ITER"
Write-Host "[run_server] ENERGY_GATE       = $env:SAFESIGNAL_ENERGY_GATE_ENABLED"
Write-Host "[run_server] FALL_CONSECUTIVE  = $env:SAFESIGNAL_FALL_CONSECUTIVE_N"
Write-Host "[run_server] CONF_DIAG         = $env:SAFESIGNAL_CONF_DIAG"
Write-Host "[run_server] COOLDOWN_SEC      = $env:COOLDOWN_SEC"

# main.py 는 server/ 를 cwd 로 가정 (from receiver..., from dashboard... 상대 import)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

python main.py
