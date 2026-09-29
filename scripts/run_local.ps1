<#
.SYNOPSIS
    Windows equivalent of solve.sh. Runs the full pipeline and writes every graded artefact.

.DESCRIPTION
    Native Windows + CUDA, no WSL and no Docker required. Docker GPU passthrough on Windows
    needs WSL2 plus the NVIDIA Container Toolkit; if you already have that, use the
    Dockerfile instead. This script is the shorter path.

.PARAMETER ChapmanRoot
    Folder holding the WFDB records. Defaults to $env:CHAPMAN_ROOT.

.PARAMETER Smoke
    2 + 1 epochs on the real corpus. Checks wiring only; the resulting metrics are
    meaningless and the artefact tests will correctly fail their thresholds.

.PARAMETER Stage
    Run one stage only: verify, train, evaluate, explain, stats. Default: all of them.

.EXAMPLE
    .\scripts\run_local.ps1 -ChapmanRoot "C:\H Research Projects\WFDB_ChapmanShaoxing" -Stage verify

.EXAMPLE
    .\scripts\run_local.ps1 -ChapmanRoot "C:\H Research Projects\WFDB_ChapmanShaoxing"
#>

[CmdletBinding()]
param(
    [string] $ChapmanRoot     = $env:CHAPMAN_ROOT,
    [string] $OutputDir       = "",
    [ValidateSet("all","verify","train","evaluate","explain","stats")]
    [string] $Stage           = "all",
    [ValidateSet("canonical_v2","canonical_hmgmedformer","rhythm_first","paper_reported")]
    [string] $ResolutionOrder = "canonical_v2",
    [ValidateSet("v2","paper")]
    [string] $Recipe          = "v2",
    [int]    $PretrainEpochs  = 40,
    [int]    $LoraEpochs      = 15,
    [int]    $BatchSize       = 64,
    [int]    $NumWorkers      = 8,
    [string] $Device          = "auto",
    [switch] $Smoke
)

$ErrorActionPreference = "Stop"

# scripts/ sits at the repository root; the task package is four levels below tasks/.
$RepoRoot    = Split-Path -Parent $PSScriptRoot
$TaskDir     = Join-Path $RepoRoot "tasks\medical-ai\ecg-arrhythmia\lightweight-explainable-lora-vit"
$SolutionDir = Join-Path $TaskDir "solution"
$SolutionSrc = Join-Path $SolutionDir "src"
if (-not $OutputDir) { $OutputDir = Join-Path $TaskDir "outputs" }

$env:PYTHONPATH   = "$SolutionSrc;$env:PYTHONPATH"
$env:DATA_DIR     = Join-Path $TaskDir "environment\data"
$env:OUTPUT_DIR   = $OutputDir
$env:MPLBACKEND   = "Agg"
$env:PYTHONHASHSEED = "0"

function Write-Step($msg) { Write-Host "`n[run] $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "`n[run] ERROR: $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
if (-not $ChapmanRoot) {
    Fail @"
No corpus path given.

    .\scripts\run_local.ps1 -ChapmanRoot "C:\H Research Projects\WFDB_ChapmanShaoxing"

or set `$env:CHAPMAN_ROOT. The corpus is open access but is not vendored in this
repository; see environment\data\README.md.
"@
}
if (-not (Test-Path -LiteralPath $ChapmanRoot)) { Fail "corpus folder not found: $ChapmanRoot" }

Write-Step "checking torch and GPU"
$gpuCheck = @'
import sys, torch
print(f"  torch        {torch.__version__}  (CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    print("  WARNING: no CUDA device visible. On CPU this will take days.", file=sys.stderr)
    sys.exit(0)
p = torch.cuda.get_device_properties(0)
cap = f"sm_{p.major}{p.minor}"
archs = torch.cuda.get_arch_list()
print(f"  gpu          {p.name} ({cap}, {p.total_memory/1024**3:.1f} GB)")
print(f"  build archs  {archs}")
if not any(cap in a for a in archs):
    print(f"\n  ERROR: this torch build has no kernels for {cap}.", file=sys.stderr)
    print("  RTX 50-series (Blackwell) needs a cu128 or newer wheel:", file=sys.stderr)
    print("    pip install -r environment\\requirements-cu128.txt", file=sys.stderr)
    sys.exit(1)
'@
$gpuCheck | python -
if ($LASTEXITCODE -ne 0) { Fail "torch cannot drive this GPU; see the message above" }

if ($Smoke) {
    Write-Step "SMOKE: 2 + 1 epochs. Wiring check only; the metrics are meaningless."
    $PretrainEpochs = 2
    $LoraEpochs     = 1
}

$common = @(
    "--chapman-root",     $ChapmanRoot,
    "--output-dir",       $OutputDir,
    "--data-dir",         $env:DATA_DIR,
    "--resolution-order", $ResolutionOrder,
    "--recipe",           $Recipe,
    "--device",           $Device,
    "--batch-size",       $BatchSize,
    "--num-workers",      $NumWorkers
)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Invoke-Stage($name, $extraArgs) {
    Write-Step $name
    $argv = @("-m", "ecgvit.cli") + $common + $extraArgs
    & python @argv
    if ($LASTEXITCODE -ne 0) { Fail "stage '$name' failed with exit code $LASTEXITCODE" }
}

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
$started = Get-Date

if ($Stage -in @("all","verify"))   { Invoke-Stage "verifying corpus" @("verify-data") }
if ($Stage -in @("all","train"))    {
    Invoke-Stage "training (stage 1: $PretrainEpochs epochs full fine-tune; stage 2: $LoraEpochs epochs LoRA r=8)" `
        @("--pretrain-epochs", $PretrainEpochs, "--lora-epochs", $LoraEpochs, "train")
}
if ($Stage -in @("all","evaluate")) { Invoke-Stage "evaluating on the held-out test split" @("evaluate") }
if ($Stage -in @("all","explain"))  { Invoke-Stage "explainability: Grad-CAM, IG, SHAP, faithfulness, t-SNE" @("explain") }
if ($Stage -in @("all","stats"))    { Invoke-Stage "statistics: McNemar and DeLong (LoRA vs no-LoRA)" @("stats") }

$elapsed = (Get-Date) - $started
Write-Step ("done in {0:hh\:mm\:ss}. Artefacts in {1}" -f $elapsed, $OutputDir)

if ($Stage -eq "all") {
    Write-Host "`nGrade the run with:" -ForegroundColor Cyan
    Write-Host "  `$env:OUTPUT_DIR = `"$OutputDir`""
    Write-Host "  `$env:CHAPMAN_ROOT = `"$ChapmanRoot`""
    Write-Host "  python -m pytest tests -m `"artifacts or corpus`""
}
