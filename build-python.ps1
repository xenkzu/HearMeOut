# Build Python FastAPI sidecar for Tauri
python -m PyInstaller --onefile --distpath src-tauri --name api src-python/main.py

# Rename to match target triple
$triple = "x86_64-pc-windows-msvc"
Move-Item -Force "src-tauri/api.exe" "src-tauri/api-$triple.exe"

Write-Host "Python sidecar built and renamed successfully!" -ForegroundColor Green
