<#
.SYNOPSIS
  One-shot environment setup for UnReflect Batch (Windows).

.DESCRIPTION
  Creates a project .venv, installs a CUDA-enabled PyTorch (Blackwell/sm_120 safe),
  installs the pinned dependencies (incl. the exact `transformers` commit the
  checkpoint needs), optionally installs the Streamlit GUI, and downloads the
  pretrained weights (~5.9 GB).

  Idempotent: re-running is safe and fast once things are installed.

.PARAMETER Gui        Also install Streamlit (for run_app.bat).
.PARAMETER SkipWeights Skip the `unreflectanything download --weights` step.
.PARAMETER CudaIndex  PyTorch wheel index (default cu128; use cu130 to match a very new driver).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Gui
#>
param(
    [switch]$Gui,
    [switch]$SkipWeights,
    [string]$CudaIndex = "https://download.pytorch.org/whl/cu128"
)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent $PSScriptRoot
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"

function Write-Step($m) { Write-Host "`n==== $m ====" -ForegroundColor Cyan }

# --- 1. Find a Python 3.11+ interpreter to build the venv -------------------
function Resolve-BasePython {
    $cands = @(
        @("py", @("-3.11")),
        @("py", @("-3.12")),
        @("py", @("-3")),
        @("python", @())
    )
    foreach ($c in $cands) {
        try {
            $v = & $c[0] @($c[1]) -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $v) {
                $p = $v.Trim().Split(".")
                if ([int]$p[0] -ge 3 -and [int]$p[1] -ge 11) {
                    return @{ exe = $c[0]; args = $c[1]; ver = $v.Trim() }
                }
            }
        } catch { }
    }
    throw "No Python 3.11+ found. Install it from https://www.python.org/downloads/ (check 'Add to PATH')."
}

if (-not (Test-Path $VenvPy)) {
    Write-Step "Creating virtual environment (.venv)"
    $base = Resolve-BasePython
    Write-Host "Using base Python $($base.ver) via '$($base.exe) $($base.args)'"
    & $base.exe @($base.args) -m venv (Join-Path $Root ".venv")
} else {
    Write-Step "Reusing existing .venv"
}

Write-Step "Upgrading pip / setuptools / wheel"
& $VenvPy -m pip install --upgrade pip setuptools wheel

# --- 2. PyTorch from the CUDA index (CPU-only on default PyPI for Windows!) --
Write-Step "Installing CUDA PyTorch ($CudaIndex)"
& $VenvPy -m pip install torch==2.9.1 torchvision==0.24.1 --index-url $CudaIndex

Write-Step "Verifying CUDA / sm_120"
& $VenvPy -c @"
import torch
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())
try:
    print('arch_list', torch.cuda.get_arch_list())
    if torch.cuda.is_available():
        print('device', torch.cuda.get_device_name(0), 'cap', torch.cuda.get_device_capability(0))
        a = torch.randn(256,256,device='cuda'); _ = a@a; torch.cuda.synchronize(); print('cuda matmul OK')
except Exception as e:
    print('CUDA check warning:', e)
"@

# --- 3. App dependencies (incl. the pinned transformers commit) -------------
Write-Step "Installing app dependencies (requirements.txt)"
& $VenvPy -m pip install -r (Join-Path $Root "requirements.txt")

if ($Gui) {
    Write-Step "Installing Streamlit GUI"
    & $VenvPy -m pip install "streamlit>=1.40"
}

# --- 4. Pretrained weights (~5.9 GB) ---------------------------------------
if (-not $SkipWeights) {
    Write-Step "Downloading pretrained weights (~5.9 GB) — one time"
    $ura = Join-Path $Root ".venv\Scripts\unreflectanything.exe"
    & $ura download --weights
    Write-Step "Verifying weights load"
    & $ura verify --weights
} else {
    Write-Host "`nSkipping weights download (-SkipWeights). Run later:" -ForegroundColor Yellow
    Write-Host "    .\.venv\Scripts\unreflectanything.exe download --weights"
}

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "  CLI : .\.venv\Scripts\python.exe main.py --input <in> --output <out> --recursive --make-preview"
if ($Gui) { Write-Host "  GUI : run_app.bat" }
