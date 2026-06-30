# Lanza la app localmente y abre el navegador. Uso:  .\run.ps1
# Cortá con Ctrl+C cuando termines.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "No encuentro el venv (.venv). Crealo con:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -e ." -ForegroundColor Yellow
    exit 1
}

$url = "http://127.0.0.1:8000"
Write-Host "Abriendo $url en el navegador (esperando que levante el server)..." -ForegroundColor Cyan
# Abre el navegador recién cuando /health responde, así no ves una pagina rota.
Start-Job {
    param($u)
    for ($i = 0; $i -lt 30; $i++) {
        try { if ((Invoke-WebRequest "$u/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { Start-Process $u; break } } catch {}
        Start-Sleep -Milliseconds 800
    }
} -ArgumentList $url | Out-Null

Write-Host "Servidor en $url  (Ctrl+C para cortar)" -ForegroundColor Green
& $py -m uvicorn markowitz_optimizer.api.main:app --port 8000
