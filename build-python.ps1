# Build Python FastAPI sidecar for Tauri
python -m PyInstaller --onefile --distpath src-tauri --name fastapi-backend src-python/main.py

# Rename to match target triple and move to binaries folder
$triple = "x86_64-pc-windows-msvc"
if (!(Test-Path "src-tauri/binaries")) { New-Item -ItemType Directory "src-tauri/binaries" }
Move-Item -Force "src-tauri/fastapi-backend.exe" "src-tauri/binaries/fastapi-backend-$triple.exe"

Write-Host "Python sidecar built and renamed successfully!" -ForegroundColor Green
