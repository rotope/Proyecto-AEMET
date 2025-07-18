# Script para empaquetar

Write-Host " Empaquetando Lambda Function AEMET..." -ForegroundColor Green

Set-Location "../lambda_functions"

if (-not (Test-Path "lambda_aemet.py")) {
    Write-Host " Error: No se encuentra lambda_aemet.py" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "requirements.txt")) {
    Write-Host " Error: No se encuentra requirements.txt" -ForegroundColor Red
    exit 1
}

Write-Host " Archivos encontrados:" -ForegroundColor Green
Write-Host "  - lambda_aemet.py" -ForegroundColor White
Write-Host "  - requirements.txt" -ForegroundColor White

# Crear directorio temporal
$tempDir = "lambda_package"
if (Test-Path $tempDir) { 
    Write-Host " Limpiando directorio anterior..." -ForegroundColor Yellow
    Remove-Item $tempDir -Recurse -Force 
}
New-Item -ItemType Directory -Path $tempDir

Write-Host " Copiando lambda_aemet.py..." -ForegroundColor Cyan
Copy-Item "lambda_aemet.py" "$tempDir/"

# Instalar dependencias
Write-Host " Instalando dependencias..." -ForegroundColor Yellow
pip install -r requirements.txt -t $tempDir/ --quiet

# Crear archivo ZIP
Write-Host " Creando archivo ZIP..." -ForegroundColor Magenta
$zipPath = "lambda_aemet.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

Compress-Archive -Path "$tempDir/*" -DestinationPath $zipPath -Force

# Limpiar directorio
Remove-Item $tempDir -Recurse -Force

