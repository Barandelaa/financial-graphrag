from __future__ import annotations

"""App de escritorio Windows con pywebview (exe único auto-bootstrap).

- Backend: `api.py` (FastAPI + SSE + HITL + progreso + cancelación)
- Frontend: `static/index.html`
- Primer arranque: crea carpetas/data/.env solo, pide keys y chequea Ollama.

Uso dev:
    .venv\\Scripts\\python.exe desktop.py

Generar .exe (onefile): powershell -ExecutionPolicy Bypass -File build_desktop.ps1
"""

import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from src.bootstrap import (
    OLLAMA_MODEL,
    base_dir as bootstrap_base_dir,
    ollama_pull_if_possible,
    ollama_status,
    read_env_keys,
    run_bootstrap,
    single_instance_or_exit,
    write_env_keys,
)


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE_DIR = _base_dir()
try:
    os.chdir(BASE_DIR)
except Exception:
    pass
sys.path.insert(0, str(BASE_DIR))


def find_free_port(preferred: int) -> int:
    for port in [preferred] + [preferred + i for i in range(1, 20)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No hay puertos libres (8000-8019).")


def wait_for_health(port: int, timeout_s: float = 300.0) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def start_server(port: int):
    import uvicorn

    import api  # noqa: E402

    config = uvicorn.Config(
        api.app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        reload=False,
    )
    server = uvicorn.Server(config)
    start_server.instance = server  # type: ignore[attr-defined]
    server.run()


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")


def _setup_html(keys: dict, ol: dict, root: str) -> str:
    if ol["installed"] and ol["serving"] and ol["model_ok"]:
        ol_line = f"✅ Ollama OK ({_esc(OLLAMA_MODEL)}): {_esc(ol['detail'])}"
    elif ol["installed"]:
        ol_line = (f"⚠️ Ollama instalado pero falta modelo/servidor. {_esc(ol['detail'])}"
                   "<br>Pulsa «Descargar modelo» o se intentará solo al continuar.")
    else:
        ol_line = ("❌ Ollama no instalado. La app arranca igual (modo Groq limitado), "
                   "pero para uso local instálalo: <b>https://ollama.com/download</b> "
                   "y luego <b>ollama pull qwen3:8b</b>.")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body{{font-family:Segoe UI,Arial;background:#111;color:#eee;margin:0;padding:24px}}
.card{{max-width:640px;margin:auto;background:#1c1c1c;border:1px solid #333;border-radius:12px;padding:24px}}
h1{{font-size:20px;margin:0 0 4px}}p.sub{{color:#aaa;margin:0 0 16px;font-size:13px}}
label{{display:block;margin:12px 0 4px;font-size:13px;color:#ccc}}
input{{width:100%;padding:10px;border-radius:8px;border:1px solid #444;background:#222;color:#fff;box-sizing:border-box}}
button{{margin:14px 8px 0 0;padding:10px 16px;border-radius:8px;border:0;cursor:pointer;font-weight:600}}
.primary{{background:#2ea043;color:#fff}}.ghost{{background:#333;color:#fff}}
.link{{background:transparent;color:#58a6ff;text-decoration:underline;padding:10px 4px}}
.status{{margin-top:16px;font-size:13px;background:#222;border:1px solid #333;border-radius:8px;padding:12px;line-height:1.6}}
small{{color:#888}}
</style></head><body><div class="card">
<h1>Financial GraphRAG — primer arranque</h1>
<p class="sub">Carpeta de datos: {_esc(root)}<br>Todo se crea solo. Solo pega tus keys si las tienes.</p>
<label>FINNHUB_API_KEY (cotizaciones)</label>
<input id="finnhub" type="text" value="{_esc(keys.get('FINNHUB_API_KEY',''))}" placeholder="Pega tu key de finnhub.io">
<label>HF_TOKEN (embeddings bge-m3)</label>
<input id="hf" type="text" value="{_esc(keys.get('HF_TOKEN',''))}" placeholder="hf_...">
<label>GROQ_API_KEY (opcional, respaldo si no hay Ollama)</label>
<input id="groq" type="text" value="{_esc(keys.get('GROQ_API_KEY',''))}" placeholder="gsk_...">
<div><button class="primary" onclick="save()">Guardar y continuar</button>
<button class="ghost" onclick="skip()">Continuar sin keys</button></div>
<div><button class="link" onclick="pull()">Descargar modelo Ollama ({_esc(OLLAMA_MODEL)})</button>
<button class="link" onclick="dl()">Descargar Ollama</button></div>
<div class="status">{ol_line}<br><span id="msg"></span></div>
<small>Sin keys la app abre igual pero limitada. Puedes editar .env junto al .exe y reiniciar.</small>
<script>
function vals(){{return {{FINNHUB_API_KEY:document.getElementById('finnhub').value,
 HF_TOKEN:document.getElementById('hf').value, GROQ_API_KEY:document.getElementById('groq').value}}}}
function save(){{pywebview.api.save(vals()).then(()=>pywebview.api.done());}}
function skip(){{pywebview.api.done();}}
function pull(){{document.getElementById('msg').innerText='Descargando modelo, mira la consola...';
 pywebview.api.pull().then(r=>document.getElementById('msg').innerText=r);}}
function dl(){{pywebview.api.open_ollama();}}
</script></div></body></html>"""


class _SetupApi:
    def __init__(self, root: Path):
        self._root = root
        self._finished = threading.Event()

    def save(self, keys):
        try:
            write_env_keys(self._root, dict(keys or {}))
            return True
        except Exception as exc:
            return f"Error guardando .env: {exc}"

    def pull(self):
        ok, log = ollama_pull_if_possible(OLLAMA_MODEL)
        return ("Modelo listo. Ya puedes continuar." if ok
                else f"No se pudo descargar todavía. Detalle: {(log or '')[-500:]}")

    def open_ollama(self):
        webbrowser.open("https://ollama.com/download")
        return True

    def done(self):
        try:
            import webview
            for w in list(webview.windows):
                try:
                    if "configuraci" in (w.title or ""):
                        w.destroy()
                        break
                except Exception:
                    pass
        except Exception:
            pass
        self._finished.set()
        return True


def maybe_show_setup(root: Path) -> None:
    import webview

    keys = read_env_keys(root)
    ol = ollama_status()
    needs_keys = not keys.get("FINNHUB_API_KEY") and not keys.get("HF_TOKEN")
    needs_ollama = not (ol["installed"] and ol["serving"] and ol["model_ok"])
    if not needs_keys and not needs_ollama:
        return
    api = _SetupApi(root)
    win = webview.create_window(
        "Financial GraphRAG — configuración",
        html=_setup_html(keys, ol, str(root)),
        js_api=api,
        width=700, height=720,
    )
    try:
        win.events.closed += lambda: api._finished.set()  # type: ignore[attr-defined]
    except Exception:
        pass
    webview.start(gui="edgechromium", debug=False)


def main() -> int:
    try:
        import webview  # pywebview
    except ImportError:
        print("Falta pywebview. Instala con: pip install pywebview")
        return 1

    # 1) Bootstrap: crea carpetas, companies.json y .env plantilla sin preguntar.
    root = bootstrap_base_dir()
    try:
        os.chdir(root)
    except Exception:
        pass
    report = run_bootstrap()
    print(f"Datos en: {report['base_dir']}")
    if report.get("dirs_created"):
        print(f"Carpetas creadas: {report['dirs_created']}")
    if report.get("companies") != "ok":
        print(f"companies.json: {report['companies']}")
    if report.get("env") != "ok":
        print(".env plantilla creada junto al .exe.")

    # 2) Evita doble instancia (Kuzu un solo escritor).
    preferred = single_instance_or_exit()

    # 3) Ventana de setup solo si falta algo (keys u Ollama).
    maybe_show_setup(root)

    # 4) Si Ollama existe pero falta el modelo, pull automático en 2º plano.
    ol = ollama_status()
    if ol["installed"] and not ol["model_ok"]:
        print(f"Ollama: {ol['detail']} -> intentando `ollama pull {OLLAMA_MODEL}` en segundo plano...")
        threading.Thread(
            target=ollama_pull_if_possible, args=(OLLAMA_MODEL,), daemon=True
        ).start()
    elif not ol["installed"]:
        print("Ollama no instalado. La app sigue (respaldo Groq si hay key). "
              "Instala desde https://ollama.com/download y `ollama pull qwen3:8b`.")

    port = find_free_port(preferred)
    url = f"http://127.0.0.1:{port}/"

    t = threading.Thread(target=start_server, args=(port,), daemon=True)
    t.start()

    print(f"Arrancando backend en {url} ...")
    print("Esto puede tardar 1-3 min la primera vez (carga bge-m3 + reranker + qwen3:8b).")
    if not wait_for_health(port):
        print("ERROR: el backend no respondió a /health. Revisa Ollama (.env) y los logs de arriba.")
        print("Consejo: abre otra terminal y prueba `ollama list` / `ollama pull qwen3:8b`.")
        try:
            srv = getattr(start_server, "instance", None)
            if srv is not None:
                srv.should_exit = True
        except Exception:
            pass
        return 1

    print("Backend listo. Abriendo ventana...")
    window = webview.create_window(
        "Financial GraphRAG",
        url,
        width=int(os.getenv("DESKTOP_WIDTH", "1220")),
        height=int(os.getenv("DESKTOP_HEIGHT", "820")),
        min_size=(900, 600),
    )

    def _on_closed():
        try:
            srv = getattr(start_server, "instance", None)
            if srv is not None:
                srv.should_exit = True
        except Exception:
            pass

    try:
        window.events.closed += _on_closed  # type: ignore[attr-defined]
    except Exception:
        pass

    webview.start(gui="edgechromium", debug=bool(os.getenv("DESKTOP_DEBUG")))
    _on_closed()
    time.sleep(1.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
