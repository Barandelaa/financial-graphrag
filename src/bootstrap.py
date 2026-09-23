from __future__ import annotations

"""Auto-bootstrap para el .exe onefile.

Garantiza que con solo doble-click la app funcione:
  1. Crea carpetas necesarias (data/..., logs/)
  2. Restaura data/companies.json desde el bundle si falta
  3. Garantiza .env (si falta -> la UI de setup lo pide, aquí solo plantilla)
  4. Chequea Ollama: servidor + modelo; si el binario existe hace `ollama pull` solo
  5. Lock de instancia única (Kuzu = un solo escritor)

No instala nada silenciosamente salvo `ollama pull` (decisión del usuario).
Ollama y WebView2, si no existen, se explican con enlaces de descarga.
"""

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Ficheros empaquetados dentro del exe (via _MEIPASS) que se restauran junto al exe.
BUNDLED_FILES = {
    "data/companies.json": b'{\n  "companies": ["AAPL"],\n  "default_years": [2024]\n}\n',
}

REQUIRED_DIRS = [
    "data",
    "data/graph",
    "data/vector_store",
    "data/processed_chunks",
    "data/raw_10k",
    "data/sec",
    "logs",
]

ENV_TEMPLATE = (
    "# Generado automáticamente por FinancialGraphRAG.\n"
    "# Pega tus keys y reinicia la app.\n"
    "FINNHUB_API_KEY=\n"
    "HF_TOKEN=\n"
    "GROQ_API_KEY=\n"
)


def base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def bundled_path(rel: str) -> Path | None:
    """Busca un recurso empaquetado en _MEIPASS (onefile) o junto al repo (dev)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        p = Path(meipass) / rel
        if p.is_file():
            return p
    repo_file = Path(__file__).resolve().parent.parent / rel
    if repo_file.is_file():
        return repo_file
    return None


def ensure_dirs(root: Path) -> list[str]:
    created = []
    for rel in REQUIRED_DIRS:
        d = root / rel
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(rel)
    return created


def ensure_companies_json(root: Path) -> str:
    """Devuelve: 'ok' | 'restored-bundle' | 'created-default'."""
    target = root / "data" / "companies.json"
    if target.is_file():
        try:
            with open(target, encoding="utf-8") as f:
                json.load(f)
            return "ok"
        except Exception:
            bak = target.with_suffix(".json.corrupt.bak")
            try:
                shutil.copyfile(target, bak)
            except Exception:
                pass
    src = bundled_path("data/companies.json")
    try:
        if src is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
            return "restored-bundle"
    except Exception as exc:
        logger.warning("No se pudo restaurar companies.json del bundle: %s", exc)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(BUNDLED_FILES["data/companies.json"])
    return "created-default"


def ensure_env_file(root: Path) -> str:
    """Devuelve: 'ok' | 'created-template'. No pide keys aquí (lo hace la ventana setup)."""
    target = root / ".env"
    if target.is_file() and target.stat().st_size > 0:
        return "ok"
    if not target.is_file():
        try:
            src = bundled_path(".env")
            if src is not None:
                shutil.copyfile(src, target)
                return "ok"
        except Exception:
            pass
        target.write_text(ENV_TEMPLATE, encoding="utf-8")
        return "created-template"
    return "ok"


def read_env_keys(root: Path) -> dict:
    keys = {"FINNHUB_API_KEY": "", "HF_TOKEN": "", "GROQ_API_KEY": ""}
    env_file = root / ".env"
    if not env_file.is_file():
        return keys
    try:
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k in keys:
                keys[k] = v.strip().strip("\"'").strip()
    except Exception:
        pass
    # Las variables de entorno ya exportadas mandan.
    for k in keys:
        if os.getenv(k):
            keys[k] = os.getenv(k, "")
    return keys


def write_env_keys(root: Path, keys: dict) -> None:
    target = root / ".env"
    lines = ["# Guardado desde la ventana de configuración de FinancialGraphRAG."]
    for k in ("FINNHUB_API_KEY", "HF_TOKEN", "GROQ_API_KEY"):
        v = (keys.get(k) or "").strip().strip("\"'").strip()
        lines.append(f"{k}={v}")
        if v:
            os.environ[k] = v
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ollama_status(model: str = OLLAMA_MODEL, base_url: str = OLLAMA_URL) -> dict:
    """No instala nada. Solo diagnostica. Claves: installed, serving, model_ok, detail."""
    installed = shutil.which("ollama") is not None
    serving, model_ok, detail = False, False, ""
    try:
        with urllib.request.urlopen(base_url + "/api/tags", timeout=4) as r:
            data = json.load(r)
        serving = True
        names = [m.get("name", "") for m in data.get("models", [])]
        model_ok = any(n == model or n.startswith(model) for n in names)
        detail = f"Modelos: {', '.join(sorted(names)) or 'ninguno'}"
    except Exception as exc:
        detail = f"Sin servidor Ollama en {base_url}: {exc}"
    return {"installed": installed, "serving": serving, "model_ok": model_ok, "detail": detail}


def ollama_pull_if_possible(model: str = OLLAMA_MODEL) -> tuple[bool, str]:
    """Si hay binario ollama, ejecuta `ollama pull <modelo>`. Devuelve (ok, log)."""
    exe = shutil.which("ollama")
    if not exe:
        return False, "Ollama no está instalado."
    try:
        proc = subprocess.run(
            [exe, "pull", model],
            capture_output=True, text=True, timeout=1800,
        )
        log = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, log[-2000:]
    except Exception as exc:
        return False, str(exc)


def webview2_ok() -> tuple[bool, str]:
    """Chequeo best-effort del runtime WebView2 en Windows."""
    if sys.platform != "win32":
        return True, "No Windows: pywebview usará el backend del sistema."
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\EdgeUpdate\ClientState\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
        ):
            return True, "WebView2 detectado en registro."
    except Exception:
        pass
    # No es concluyente (WebView2 suele venir con Win10/11) -> asumimos ok,
    # pywebview dará el error final con enlace si falta.
    return True, "WebView2 probablemente presente (Windows 10/11 lo incluye)."


_lock_socket: socket.socket | None = None


def single_instance_or_exit(port_env: str = "DESKTOP_PORT", default: int = 8000) -> int:
    """Evita dos instancias (Kuzu solo admite un escritor). Si el puerto preferido
    responde a /health, otra instancia está viva -> salir con mensaje."""
    preferred = int(os.getenv(port_env, str(default)))
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{preferred}/health", timeout=2) as r:
            if r.status == 200:
                print(f"Ya hay una instancia corriendo en el puerto {preferred}. Ciérrala antes de abrir otra.")
                sys.exit(2)
    except Exception:
        pass
    return preferred


def run_bootstrap() -> dict:
    """Punto de entrada idempotente. Crea todo lo creable sin red ni preguntas."""
    root = base_dir()
    try:
        os.chdir(root)
    except Exception:
        pass
    report = {
        "base_dir": str(root),
        "dirs_created": ensure_dirs(root),
        "companies": ensure_companies_json(root),
        "env": ensure_env_file(root),
        "ollama": ollama_status(),
        "webview2": webview2_ok(),
    }
    logger.info("Bootstrap: %s", report)
    return report
