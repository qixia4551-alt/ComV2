#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Console - Backend Server
Provides static file serving, local model scanning, and ComfyUI API proxy.
No external dependencies required - uses only Python standard library.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
import urllib.error
import mimetypes
import posixpath
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# ============================================================
# Configuration
# ============================================================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8501
COMFYUI_DEFAULT_URL = "http://127.0.0.1:8189"

# Server-side shared data store (password, templates, gallery, selections, etc.)
# Stored on THIS computer so every device (phone / other PC) reads the same data.
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "console_data.json")


def load_store():
    """Load the shared key/value data from disk."""
    try:
        if os.path.isfile(DATA_FILE):
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}


def save_store(data):
    """Persist the shared key/value data to disk."""
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


# ------------------------------------------------------------
# Login password (stored in a plain text file on THIS computer).
# Change the password by editing this file only.
# ------------------------------------------------------------
PASSWORD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "password.txt")


def get_password():
    """Read the login password from password.txt (creates it if missing)."""
    try:
        if os.path.isfile(PASSWORD_FILE):
            with open(PASSWORD_FILE, "r", encoding="utf-8") as f:
                pw = f.read().strip()
                return pw if pw else "admin"
    except Exception:
        pass
    # Create with a default so the file is easy to find and edit
    try:
        with open(PASSWORD_FILE, "w", encoding="utf-8") as f:
            f.write("admin")
    except Exception:
        pass
    return "admin"


# ------------------------------------------------------------
# ComfyUI backend URL auto-detection.
# Tries an optional override file, then common ports (Comfy Desktop 8189,
# manual install 8188). Result is cached and re-probed on failure.
# ------------------------------------------------------------
COMFYUI_URL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comfyui_url.txt")
_comfy_url_cache = {"url": None}


def _comfy_candidates():
    cands = []
    # 1) optional manual override file
    try:
        if os.path.isfile(COMFYUI_URL_FILE):
            with open(COMFYUI_URL_FILE, "r", encoding="utf-8") as f:
                u = f.read().strip()
                if u:
                    if not u.startswith("http"):
                        u = "http://" + u
                    cands.append(u.rstrip("/"))
    except Exception:
        pass
    # 2) common defaults
    for u in ["http://127.0.0.1:8189", "http://127.0.0.1:8188", "http://127.0.0.1:8000"]:
        if u not in cands:
            cands.append(u)
    return cands


def _probe_comfy(url):
    try:
        req = urllib.request.Request(url.rstrip("/") + "/system_stats")
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def resolve_comfy_url(force=False):
    """Return a working ComfyUI base URL, probing candidates as needed."""
    if _comfy_url_cache["url"] and not force:
        return _comfy_url_cache["url"]
    for u in _comfy_candidates():
        if _probe_comfy(u):
            _comfy_url_cache["url"] = u
            return u
    # nothing responded; keep default so errors are meaningful
    _comfy_url_cache["url"] = None
    return COMFYUI_DEFAULT_URL


# ------------------------------------------------------------
# Model architecture detection (SDXL / SD1.5 / SD2 / other).
# Reads only the safetensors header (fast) and inspects a cross-attention
# to_k weight's context dimension: 2048=SDXL, 768=SD1.5, 1024=SD2.
# ------------------------------------------------------------
_arch_cache = {}


def _read_safetensors_header(fp):
    import struct
    with open(fp, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8", "replace"))


def detect_arch(fp):
    """Return 'SDXL' / 'SD1.5' / 'SD2' / 'other'. Cached by path+mtime."""
    try:
        key = fp + "|" + str(os.path.getmtime(fp))
    except Exception:
        key = fp
    if key in _arch_cache:
        return _arch_cache[key]
    arch = "other"
    try:
        hdr = _read_safetensors_header(fp)
        in_dim = None
        lora_dim = None
        ckpt_dim = None
        for k, v in hdr.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            kl = k.lower()
            # cross-attention to_k projects the text-context dim (SDXL 2048 / SD1.5 768)
            if "attn2" in kl and "to_k" in kl:
                sh = v.get("shape")
                if not sh or len(sh) < 2:
                    continue
                if "lora_down" in kl or "lora_a" in kl:
                    lora_dim = sh[-1]      # [rank, in_features]
                elif "lora_up" in kl or "lora_b" in kl:
                    continue               # [out, rank] — not the context dim
                elif kl.endswith("to_k.weight") or "lora" not in kl:
                    ckpt_dim = sh[-1]      # [inner, context]
        in_dim = lora_dim if lora_dim is not None else ckpt_dim
        if in_dim == 2048:
            arch = "SDXL"
        elif in_dim == 768:
            arch = "SD1.5"
        elif in_dim == 1024:
            arch = "SD2"
        elif in_dim is not None:
            arch = "ctx" + str(in_dim)
    except Exception:
        arch = "unknown"
    _arch_cache[key] = arch
    return arch


def scan_arch(models_dir):
    """Return {'checkpoints': {name: arch}, 'loras': {name: arch}}."""
    out = {"checkpoints": {}, "loras": {}}
    if not models_dir or not os.path.isdir(models_dir):
        return out
    mapping = {"checkpoints": ["checkpoints", "checkpoint"], "loras": ["loras", "lora"]}
    for cat, subs in mapping.items():
        for sub in subs:
            base = os.path.join(models_dir, sub)
            if not os.path.isdir(base):
                continue
            for root, _dirs, files in os.walk(base):
                for fn in files:
                    if fn.lower().endswith(".safetensors"):
                        rel = os.path.relpath(os.path.join(root, fn), models_dir).replace("\\", "/")
                        out[cat][rel] = detect_arch(os.path.join(root, fn))
    return out

# Try to find ComfyUI models directory automatically
COMFYUI_MODELS_DIR = None

# Common ComfyUI installation paths to search
_SEARCH_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ComfyUI", "models"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ComfyUI", "models"),
    "C:\\ComfyUI\\models",
    "D:\\ComfyUI\\models",
    "E:\\ComfyUI\\models",
    # Comfy Desktop paths
    "D:\\Comfy-Desktop\\ComfyUI-Shared\\models",
    "C:\\Comfy-Desktop\\ComfyUI-Shared\\models",
    "E:\\Comfy-Desktop\\ComfyUI-Shared\\models",
    os.path.expanduser("~/ComfyUI/models"),
    os.path.expanduser("~/stable-diffusion-webui/models"),
]

# Also try reading Comfy Desktop settings for model path
def _find_comfy_desktop_models():
    """Try to find model directory from Comfy Desktop settings."""
    import glob
    # Common AppData paths for Comfy Desktop
    appdata = os.environ.get("APPDATA", "")
    candidates = [
        os.path.join(appdata, "Comfy Desktop", "settings.json"),
        os.path.join(appdata, "Comfy", "settings.json"),
    ]
    for settings_path in candidates:
        if os.path.isfile(settings_path):
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                dirs = settings.get("modelsDirs", [])
                if dirs and isinstance(dirs, list) and len(dirs) > 0:
                    return dirs[0]
            except Exception:
                pass
    return None

_comfy_desktop_dir = _find_comfy_desktop_models()
if _comfy_desktop_dir:
    _SEARCH_PATHS.insert(0, _comfy_desktop_dir)

for _p in _SEARCH_PATHS:
    _p = os.path.abspath(_p)
    if os.path.isdir(_p):
        # Check if it has checkpoints subdirectory
        if os.path.isdir(os.path.join(_p, "checkpoints")) or os.path.isdir(os.path.join(_p, "vae")):
            COMFYUI_MODELS_DIR = _p
            break

# Model subdirectories to scan
MODEL_CATEGORIES = {
    "checkpoints": ["checkpoints", "checkpoint", "ckpt", "safetensors"],
    "vae": ["vae", "VAE"],
    "loras": ["loras", "lora", "LoRA"],
}

# Supported model file extensions
MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}


def scan_models(models_dir):
    """Scan model directories and return categorized model list."""
    result = {"checkpoints": [], "vae": [], "loras": []}

    if not models_dir or not os.path.isdir(models_dir):
        return result

    for category, subdirs in MODEL_CATEGORIES.items():
        found = set()
        for subdir in subdirs:
            dir_path = os.path.join(models_dir, subdir)
            if os.path.isdir(dir_path):
                for root, dirs, files in os.walk(dir_path):
                    for f in files:
                        ext = os.path.splitext(f)[1].lower()
                        if ext in MODEL_EXTENSIONS:
                            # Get relative path from models dir
                            rel_path = os.path.relpath(os.path.join(root, f), models_dir)
                            rel_path = rel_path.replace("\\", "/")
                            found.add(rel_path)
        result[category] = sorted(list(found))

    return result


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server for handling multiple concurrent requests."""
    daemon_threads = True
    allow_reuse_address = True


class ConsoleHandler(BaseHTTPRequestHandler):
    """HTTP request handler for ComfyUI Console."""

    server_version = "ComfyUIConsole/1.0"

    def log_message(self, format, *args):
        """Override to provide cleaner logging."""
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), format % args))

    # --------------------------------------------------------
    # Utility methods
    # --------------------------------------------------------
    def _send_json(self, data, status=200):
        """Send JSON response."""
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message, status=500):
        """Send error JSON response."""
        self._send_json({"error": message, "success": False}, status)

    def _read_body(self):
        """Read request body as JSON."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _serve_file(self, filepath):
        """Serve a static file."""
        try:
            with open(filepath, "rb") as f:
                content = f.read()
        except (IOError, OSError):
            self.send_error(404, "File Not Found")
            return

        content_type = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def _proxy_comfyui(self, method, path, body=None, comfyui_url=None):
        """Proxy a request to ComfyUI."""
        if not comfyui_url:
            comfyui_url = COMFYUI_DEFAULT_URL

        url = comfyui_url.rstrip("/") + "/" + path.lstrip("/")

        try:
            data = None
            headers = {"Content-Type": "application/json"}

            if body and method in ("POST", "PUT"):
                data = json.dumps(body).encode("utf-8")
                headers["Content-Length"] = str(len(data))

            req = urllib.request.Request(url, data=data, method=method, headers=headers)

            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_body = resp.read()
                content_type = resp.headers.get("Content-Type", "application/json")

                self.send_response(resp.status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(resp_body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                # Generated images are immutable (unique filename per generation),
                # so let the browser cache them aggressively. This stops 4K images
                # from being re-downloaded every time they are opened.
                low = (path or "").lower()
                if low.startswith("view") or "/view" in low or content_type.startswith("image/"):
                    self.send_header("Cache-Control", "public, max-age=31536000, immutable")
                self.end_headers()
                self.wfile.write(resp_body)
                return True

        except urllib.error.URLError as e:
            # The cached ComfyUI URL may be stale (ComfyUI restarted / different
            # port). Invalidate so the next request re-probes for a live backend.
            _comfy_url_cache["url"] = None
            self._send_error_json(f"ComfyUI connection failed: {str(e.reason)}", 502)
            return False
        except Exception as e:
            self._send_error_json(f"Proxy error: {str(e)}", 500)
            return False

    # --------------------------------------------------------
    # CORS preflight
    # --------------------------------------------------------
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    # --------------------------------------------------------
    # GET requests
    # --------------------------------------------------------
    def do_GET(self):
        """Handle GET requests."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # Generic ComfyUI passthrough proxy (enables remote / tunnel access):
        # the browser talks to same-origin /comfy/*, this server forwards it to the
        # local ComfyUI at 127.0.0.1:8189. Query string is preserved.
        if path == "/comfy" or path.startswith("/comfy/"):
            sub = self.path[len("/comfy"):]  # keep leading slash + query string
            self._proxy_comfyui("GET", sub, comfyui_url=resolve_comfy_url())
            return

        # API: Get local models
        if path == "/api/models":
            models_dir = query.get("dir", [COMFYUI_MODELS_DIR])[0]
            if not models_dir:
                models_dir = COMFYUI_MODELS_DIR
            models = scan_models(models_dir)
            self._send_json({
                "success": True,
                "models": models,
                "models_dir": models_dir or "",
                "found": bool(models_dir and os.path.isdir(models_dir))
            })
            return

        # API: Read the shared server-side data store (all devices share this)
        if path == "/api/store":
            self._send_json({"success": True, "data": load_store()})
            return

        # API: Verify login password (password lives in password.txt on this PC)
        if path == "/api/auth":
            pw = query.get("pw", [""])[0]
            self._send_json({"success": True, "ok": (pw == get_password())})
            return

        # API: Model architecture map (for LoRA/checkpoint compatibility hints)
        if path == "/api/arch":
            models_dir = query.get("dir", [COMFYUI_MODELS_DIR])[0] or COMFYUI_MODELS_DIR
            self._send_json({"success": True, "arch": scan_arch(models_dir)})
            return

        # API: Server info
        if path == "/api/info":
            self._send_json({
                "success": True,
                "server": "ComfyUI Console",
                "version": "1.0.0",
                "models_dir": COMFYUI_MODELS_DIR or "",
                "comfyui_default": COMFYUI_DEFAULT_URL,
                "port": SERVER_PORT
            })
            return

        # API: Proxy ComfyUI object_info
        if path == "/api/object_info":
            comfyui_url = query.get("url", [COMFYUI_DEFAULT_URL])[0]
            self._proxy_comfyui("GET", "object_info", comfyui_url=comfyui_url)
            return

        # API: Proxy ComfyUI history
        if path == "/api/history":
            comfyui_url = query.get("url", [COMFYUI_DEFAULT_URL])[0]
            self._proxy_comfyui("GET", "history", comfyui_url=comfyui_url)
            return

        # API: Proxy ComfyUI view (image)
        if path == "/api/view":
            comfyui_url = query.get("url", [COMFYUI_DEFAULT_URL])[0]
            filename = query.get("filename", [""])[0]
            subfolder = query.get("subfolder", [""])[0]
            type_ = query.get("type", ["output"])[0]
            img_path = f"view?filename={urllib.parse.quote(filename)}&subfolder={urllib.parse.quote(subfolder)}&type={type_}"
            self._proxy_comfyui("GET", img_path, comfyui_url=comfyui_url)
            return

        # API: Proxy ComfyUI queue
        if path == "/api/queue":
            comfyui_url = query.get("url", [COMFYUI_DEFAULT_URL])[0]
            self._proxy_comfyui("GET", "queue", comfyui_url=comfyui_url)
            return

        # API: Detect local IP
        if path == "/api/ip":
            import socket
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except Exception:
                local_ip = "127.0.0.1"
            self._send_json({"success": True, "ip": local_ip, "port": SERVER_PORT})
            return

        # Static files
        if path == "/" or path == "":
            path = "/index.html"

        # Sanitize path
        clean_path = posixpath.normpath(path).lstrip("/")
        if clean_path.startswith(".."):
            self.send_error(403, "Forbidden")
            return

        # Map to local directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(base_dir, clean_path)

        if os.path.isfile(filepath):
            self._serve_file(filepath)
        else:
            # Try serving index.html for SPA routes
            index_path = os.path.join(base_dir, "index.html")
            if os.path.isfile(index_path):
                self._serve_file(index_path)
            else:
                self.send_error(404, "File Not Found")

    # --------------------------------------------------------
    # POST requests
    # --------------------------------------------------------
    def do_POST(self):
        """Handle POST requests."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        body = self._read_body()

        # Generic ComfyUI passthrough proxy (POST), for remote / tunnel access.
        if path == "/comfy" or path.startswith("/comfy/"):
            sub = self.path[len("/comfy"):]  # keep leading slash + query string
            self._proxy_comfyui("POST", sub, body=body, comfyui_url=resolve_comfy_url())
            return

        # API: Write the shared server-side data store.
        # Body: {"data": {...}} to replace the whole store, or
        #       {"key": "...", "value": ...} to set a single key.
        if path == "/api/store":
            if isinstance(body.get("data"), dict):
                ok = save_store(body["data"])
                self._send_json({"success": ok})
            elif "key" in body:
                data = load_store()
                data[body["key"]] = body.get("value")
                ok = save_store(data)
                self._send_json({"success": ok})
            else:
                self._send_error_json("Invalid store payload", 400)
            return

        # API: Generate image (proxy to ComfyUI /prompt)
        if path == "/api/generate":
            comfyui_url = body.get("comfyui_url", COMFYUI_DEFAULT_URL)
            prompt = body.get("prompt", {})
            client_id = body.get("client_id", "")

            self._proxy_comfyui("POST", "prompt", {
                "prompt": prompt,
                "client_id": client_id
            }, comfyui_url=comfyui_url)
            return

        # API: Interrupt generation
        if path == "/api/interrupt":
            comfyui_url = body.get("comfyui_url", COMFYUI_DEFAULT_URL)
            self._proxy_comfyui("POST", "interrupt", {}, comfyui_url=comfyui_url)
            return

        # API: Free memory
        if path == "/api/free":
            comfyui_url = body.get("comfyui_url", COMFYUI_DEFAULT_URL)
            self._proxy_comfyui("POST", "free", {"unload_models": True}, comfyui_url=comfyui_url)
            return

        # API: Scan custom models directory
        if path == "/api/scan":
            models_dir = body.get("dir", "")
            if models_dir and os.path.isdir(models_dir):
                models = scan_models(models_dir)
                self._send_json({
                    "success": True,
                    "models": models,
                    "models_dir": models_dir
                })
            else:
                self._send_error_json(f"Directory not found: {models_dir}", 400)
            return

        self._send_error_json("Unknown endpoint", 404)


def main():
    """Main entry point."""
    print("=" * 60)
    print("  ComfyUI Console Server v1.0")
    print("=" * 60)
    print(f"  Server Port : {SERVER_PORT}")
    print(f"  Local URL   : http://127.0.0.1:{SERVER_PORT}")
    print(f"  ComfyUI URL : {COMFYUI_DEFAULT_URL}")
    print(f"  Models Dir  : {COMFYUI_MODELS_DIR or 'Not found - auto-detecting'}")
    print("=" * 60)
    print("  Press Ctrl+C to stop the server")
    print("=" * 60)
    print()

    # Try to get local IP
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"  Network URL : http://{local_ip}:{SERVER_PORT}")
        print(f"  (Use this URL on your phone/other devices)")
        print()
    except Exception:
        pass

    server = ThreadingHTTPServer((SERVER_HOST, SERVER_PORT), ConsoleHandler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
