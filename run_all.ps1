# End-to-end pipeline: data -> training -> evaluation.
# Usage:  .\run_all.ps1 [-Epochs 18]
param([int]$Epochs = 18)
$ErrorActionPreference = "Stop"

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " GTSRB traffic sign recognition - full run"    -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
python -m src.cli info

Write-Host "`n[1/4] preparing data (downloads ~365 MB the first time)" -ForegroundColor Yellow
python -m src.cli prepare

Write-Host "`n[2/4] training for $Epochs epochs" -ForegroundColor Yellow
python -m src.cli train --epochs $Epochs

Write-Host "`n[3/4] evaluating on the official test set" -ForegroundColor Yellow
python -m src.cli evaluate

Write-Host "`n[4/4] running tests" -ForegroundColor Yellow
python -m unittest discover tests

Write-Host "`nDone. Metrics and figures are in results/" -ForegroundColor Green
