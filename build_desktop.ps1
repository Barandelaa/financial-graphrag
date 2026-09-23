# Genera dist\FinancialGraphRAG.exe (UN SOLO .exe, con consola).
# Auto-bootstrap: al ejecutar crea carpetas/data/.env solo, pide keys y chequea Ollama.
# Lo único NO empaquetable: Ollama (modelo qwen3:8b) y WebView2 del sistema.
#
# Uso:
#   powershell -ExecutionPolicy Bypass -File build_desktop.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

# Limpieza previa
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }

& ".\.venv\Scripts\pyinstaller.exe" `
  --noconfirm `
  --onefile `
  --console `
  --name "FinancialGraphRAG" `
  --add-data "static;static" `
  --add-data "src;src" `
  --add-data "api.py;." `
  --add-data "data\companies.json;data" `
  --hidden-import "uvicorn.logging" `
  --hidden-import "uvicorn.loops.auto" `
  --hidden-import "uvicorn.protocols.http.auto" `
  --hidden-import "uvicorn.protocols.websockets.auto" `
  --hidden-import "uvicorn.lifespan.on" `
  --hidden-import "webview" `
  --collect-all "api" `
  --collect-all "src" `
  --exclude-module "torch.utils.tensorboard" `
  desktop.py

Write-Host ""
Write-Host "== Listo ==" -ForegroundColor Green
Write-Host "Ejecutable: dist\FinancialGraphRAG.exe  (un solo archivo, llévalo donde quieras)"
Write-Host ""
Write-Host "Qué hace solo al doble-click:" -ForegroundColor Yellow
Write-Host " 1. Crea data\, logs\ y .env plantilla junto al .exe (si faltan)."
Write-Host " 2. Si faltan keys u Ollama, abre ventana de configuración para pegarlas."
Write-Host " 3. Si Ollama está instalado pero falta el modelo, hace 'ollama pull qwen3:8b' solo."
Write-Host ""
Write-Host "Lo único manual (no empaquetable): instalar Ollama desde https://ollama.com/download"
Write-Host "WebView2 ya viene en Win10/11. No abras dos instancias a la vez (Kuzu = un escritor)."
