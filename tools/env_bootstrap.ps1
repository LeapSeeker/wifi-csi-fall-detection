<#
  env_bootstrap.ps1 - quick environment setup after reboot.
  (This PC: only D: persists; C: is wiped on shutdown.)

  What it does:
    1) Add portable Git + Python base to this session's PATH
    2) Set execution policy Bypass for the Process scope (unblocks Activate.ps1)
    3) Activate .venv
    4) Verify git / python / torch(CUDA) / package integrity vs lock

  Usage (MUST dot-source so PATH/venv apply to the current session):
      . .\tools\env_bootstrap.ps1

  Plain run (.\tools\env_bootstrap.ps1) only affects a child process.
  See docs/ENV_SETUP.md for the Korean explanation and manual fallback.
#>

# repo root = parent of this script's folder (tools)
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Root) { $Root = (Resolve-Path "$PSScriptRoot\..").Path }

$GitCmd   = Join-Path $Root "Git\cmd"
$PyBase   = "D:\Python311"
$VenvPy   = Join-Path $Root ".venv\Scripts\python.exe"
$Activate = Join-Path $Root ".venv\Scripts\Activate.ps1"
$Lock     = Join-Path $Root "requirements.lock.txt"

Write-Host "=== [1/4] PATH ===" -ForegroundColor Cyan
$prepend = @($GitCmd, $PyBase, (Join-Path $PyBase "Scripts"))
foreach ($p in $prepend) {
    if (Test-Path $p) {
        if ($env:PATH -notlike "*$p*") { $env:PATH = "$p;$env:PATH" }
        Write-Host "  + $p"
    } else {
        Write-Host "  ! missing: $p" -ForegroundColor Yellow
    }
}

Write-Host "=== [2/4] ExecutionPolicy (Process=Bypass) ===" -ForegroundColor Cyan
try { Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force } catch {}

Write-Host "=== [3/4] activate .venv ===" -ForegroundColor Cyan
if (Test-Path $Activate) {
    & $Activate
    Write-Host "  activated: $Activate"
} else {
    Write-Host "  ! Activate.ps1 missing - venv needs recreation (see doc)" -ForegroundColor Yellow
}

Write-Host "=== [4/4] verify ===" -ForegroundColor Cyan
& "$GitCmd\git.exe" --version
if (Test-Path $VenvPy) {
    & $VenvPy --version
    & $VenvPy -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
    if (Test-Path $Lock) {
        $lockPkgs = (Get-Content $Lock | Where-Object { $_ -match '==' } | Sort-Object)
        $freeze   = (& $VenvPy -m pip freeze | Where-Object { $_ -match '==' } | Sort-Object)
        $d = Compare-Object $lockPkgs $freeze
        if ($d) {
            Write-Host "  ! package mismatch - reinstall recommended:" -ForegroundColor Yellow
            Write-Host "      $VenvPy -m pip install -r `"$Lock`""
            $d | Format-Table -AutoSize
        } else {
            Write-Host "  OK: venv == requirements.lock.txt ($($lockPkgs.Count) pkgs)" -ForegroundColor Green
        }
    }
} else {
    Write-Host "  ! .venv missing - recreate:" -ForegroundColor Yellow
    Write-Host "      D:\Python311\python.exe -m venv `"$Root\.venv`""
    Write-Host "      `"$VenvPy`" -m pip install -r `"$Lock`""
}

Write-Host "=== done ===" -ForegroundColor Cyan
