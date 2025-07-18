# Script PowerShell para Windows
# Crear package Lambda Function de AEMET

Write-Host "Creando package de Lambda Function..." -ForegroundColor Green

# Crear directorio temporal
New-Item -ItemType Directory -Force -Path "lambda_package"
Set-Location "lambda_package"

# Copiar código de la función
Copy-Item "..\lambda_aemet.py" .

# Instalar dependencias
pip install -r ..\requirements.txt -t .

# Crear ZIP package
Compress-Archive -Path ".\*" -DestinationPath "..\lambda_aemet.zip" -Force

# Limpiar directorio temporal
Set-Location ".."
Remove-Item -Recurse -Force "lambda_package"

Write-Host "Package creado: lambda_aemet.zip" -ForegroundColor Green
Write-Host ""
Write-Host "Proximos pasos:" -ForegroundColor Yellow
Write-Host "1. Subir ZIP a AWS Lambda Console"
Write-Host "2. Configurar handler: lambda_aemet.lambda_handler"
Write-Host "3. Agregar variable AEMET_API_KEY"
Write-Host "4. Probar funcion"