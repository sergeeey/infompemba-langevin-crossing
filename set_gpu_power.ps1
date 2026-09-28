# ============================================================
# Скрипт ограничения мощности GPU для InfoMpemba
# Запускать ОТ ИМЕНИ АДМИНИСТРАТОРА!
# ============================================================

param(
    [int]$PowerLimit = 70,    # Лимит в ваттах (по умолчанию 70W ≈ 50% для RTX 5070 Ti)
    [switch]$Reset,           # Сбросить лимит (максимум)
    [switch]$Info             # Только показать информацию
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "GPU Power Limit Script для RTX 5070 Ti Laptop" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Проверяем nvidia-smi
try {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction Stop
} catch {
    Write-Host "ERROR: nvidia-smi not found!" -ForegroundColor Red
    Write-Host "Install NVIDIA drivers first." -ForegroundColor Red
    exit 1
}

# Получаем информацию о GPU
$output = & nvidia-smi --query-gpu=name,power.max_limit,temperature.gpu --format=csv,noheader,nounits
$values = $output -split ', '

$gpuName = $values[0].Trim()
$maxPower = [float]$values[1].Trim()
$currentTemp = $values[2].Trim()

Write-Host "GPU: $gpuName" -ForegroundColor Green
Write-Host "Max TDP: $maxPower W" -ForegroundColor Green
Write-Host "Current Temp: $currentTemp °C" -ForegroundColor Green
Write-Host ""

if ($Info) {
    exit 0
}

if ($Reset) {
    # Сброс к максимуму
    Write-Host "Resetting power limit to maximum ($maxPower W)..." -ForegroundColor Yellow
    & nvidia-smi -pl $maxPower
    Write-Host "Done!" -ForegroundColor Green
    exit 0
}

# Проверяем, что лимит разумный
if ($PowerLimit -gt $maxPower) {
    Write-Host "WARNING: Requested limit ($PowerLimit W) > max TDP ($maxPower W)" -ForegroundColor Yellow
    Write-Host "Setting to maximum: $maxPower W" -ForegroundColor Yellow
    $PowerLimit = [int]$maxPower
}

$powerPercent = [math]::Round($PowerLimit / $maxPower * 100)

Write-Host "Setting power limit to: $PowerLimit W ($powerPercent% of $maxPower W)" -ForegroundColor Yellow
Write-Host ""

# Устанавливаем лимит
try {
    & nvidia-smi -pl $PowerLimit 2>&1 | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: Power limit set to $PowerLimit W" -ForegroundColor Green
        Write-Host ""
        Write-Host "Expected performance:" -ForegroundColor Cyan
        Write-Host "  - GPU frequency: ~$([math]::Round(2200 * $powerPercent / 100)) MHz" -ForegroundColor White
        Write-Host "  - Temperature: 60-65°C (at $powerPercent% power)" -ForegroundColor White
        Write-Host "  - Fan noise: Low/Medium" -ForegroundColor White
        Write-Host ""
        Write-Host "Recommended experiment config:" -ForegroundColor Cyan
        
        if ($powerPercent -le 50) {
            Write-Host "  python launch_full_experiment.py --low-power" -ForegroundColor Yellow
        } else {
            Write-Host "  python launch_full_experiment.py" -ForegroundColor Yellow
        }
    } else {
        Write-Host "FAILED: Could not set power limit" -ForegroundColor Red
        Write-Host "Make sure you're running as Administrator!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Alternative: Use MSI Afterburner to set Power Limit manually" -ForegroundColor Yellow
    }
} catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
