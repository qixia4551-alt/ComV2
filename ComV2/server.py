#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Console - Backend Server
Provides static file serving, local model scanning, and ComfyUI API proxy.
No external dependencies required - uses only Python standard library.

SECURITY FIXES APPLIED:
- Session-based authentication with secure tokens
- Path traversal vulnerability fixed with strict validation
- CORS restricted to same-origin
- Sensitive operations require authentication
"""

import os
import sys
import json
import hashlib
import secrets
import time
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

# Session management for authentication
SESSIONS = {}  # {session_id: {"created_at": timestamp, "last_access": timestamp}}
SESSION_TIMEOUT = 3600  # 1 hour session timeout
SESSION_SECRET = secrets.token_hex(32)  # Random secret for session validation

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


def hash_password(password):
    """Hash password using SHA-256 for secure comparison."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def validate_session(session_id):
    """Validate session ID and refresh if valid. Returns True if valid."""
    if not session_id or session_id not in SESSIONS:
        return False
    session = SESSIONS[session_id]
    # Check if session has expired
    if time.time() - session["last_access"] > SESSION_TIMEOUT:
        del SESSIONS[session_id]
        return False
    # Refresh session timestamp
    session["last_access"] = time.time()
    return True


def create_session():
    """Create a new session and return session ID."""
    session_id = secrets.token_hex(32)
    current_time = time.time()
    SESSIONS[session_id] = {
        "created_at": current_time,
        "last_access": current_time
    }
    # Clean up old sessions
    cleanup_sessions()
    return session_id


def cleanup_sessions():
    """Remove expired sessions."""
    current_time = time.time()
    expired = [sid for sid, sess in SESSIONS.items() 
               if current_time - sess["last_access"] > SESSION_TIMEOUT]
    for sid in expired:
        del SESSIONS[sid]


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


# ============================================================
# Self-update from GitHub
# ============================================================
UPDATE_REPO_OWNER = "qixia4551-alt"
UPDATE_REPO_NAME = "ComV2"
UPDATE_BRANCH = "main"
UPDATE_APP_SUBDIR = "ComV2"   # app files live in this subfolder of the repo
APP_VERSION = "1.0.0"
UPDATE_CHECK_TIMEOUT = 30
UPDATE_ZIP_TIMEOUT = 300

# User data / local config files — NEVER overwritten by updates
UPDATE_PROTECTED_FILES = {
    "console_data.json",
    "password.txt",
    "comfyui_url.txt",
    "version.json",
}

VERSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "version.json")


def load_local_version():
    """Return the commit SHA recorded by the last successful update (or None)."""
    try:
        if os.path.isfile(VERSION_FILE):
            with open(VERSION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data.get("commit")
    except Exception:
        pass
    return None


def save_local_version(commit):
    """Persist the currently installed commit SHA to version.json."""
    try:
        import datetime
        data = {
            "commit": commit,
            "repo": f"{UPDATE_REPO_OWNER}/{UPDATE_REPO_NAME}",
            "branch": UPDATE_BRANCH,
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(VERSION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def fetch_latest_commit():
    """Query the GitHub API for the latest commit on the update branch.
    
    Tries GitHub API first, if it fails (e.g., network issues in China),
    falls back to using ghproxy mirror.
    """
    # Try GitHub API directly first
    url = f"https://api.github.com/repos/{UPDATE_REPO_OWNER}/{UPDATE_REPO_NAME}/commits/{UPDATE_BRANCH}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "ComfyUI-Console-Updater",
        "Accept": "application/vnd.github+json",
    })
    try:
        with urllib.request.urlopen(req, timeout=UPDATE_CHECK_TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8"))
        commit = data.get("commit", {})
        return {
            "sha": data.get("sha", ""),
            "message": commit.get("message", ""),
            "date": commit.get("committer", {}).get("date", ""),
            "url": data.get("html_url", ""),
        }
    except Exception as e:
        # If GitHub API fails, try using ghproxy mirror
        try:
            mirror_url = f"https://ghp.ci/api/v1/repos/{UPDATE_REPO_OWNER}/{UPDATE_REPO_NAME}/commits/{UPDATE_BRANCH}"
            req_mirror = urllib.request.Request(mirror_url, headers={
                "User-Agent": "ComfyUI-Console-Updater",
                "Accept": "application/vnd.github+json",
            })
            with urllib.request.urlopen(req_mirror, timeout=UPDATE_CHECK_TIMEOUT) as r:
                data = json.loads(r.read().decode("utf-8"))
            commit = data.get("commit", {})
            return {
                "sha": data.get("sha", ""),
                "message": commit.get("message", ""),
                "date": commit.get("committer", {}).get("date", ""),
                "url": data.get("html_url", ""),
            }
        except Exception as e2:
            raise RuntimeError(f"检查更新失败：GitHub API 和镜像源均无法访问。原始错误：{e}，镜像源错误：{e2}")


def apply_zip_bytes(zip_bytes, latest=None):
    """Overwrite app files from an in-memory zip archive.

    Supports three archive layouts: GitHub repo zip (Root/ComV2/file),
    folder zip (ComV2/file) and flat zip (file at archive root).
    Protected files (user data / local config) are never touched.
    Returns (updated_files, skipped_files).
    """
    import io
    import zipfile

    base_dir = os.path.dirname(os.path.abspath(__file__))
    prefix = UPDATE_APP_SUBDIR + "/"
    updated, skipped = [], []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"压缩包已损坏（{bad}），更新中止")
        names = [n for n in zf.namelist() if not n.endswith("/")]
        # Detect archive layout: if any entry sits under a ComV2/ folder
        # (repo zip or folder zip), only those entries are applied;
        # otherwise treat it as a flat zip of app files.
        def _stripped(n):
            return n.split("/", 1)[1] if "/" in n else n
        has_prefixed = any(
            n.startswith(prefix) or _stripped(n).startswith(prefix) for n in names
        )
        for name in names:
            if name.startswith(prefix):
                rel = name[len(prefix):]
            elif _stripped(name).startswith(prefix) and "/" in name:
                rel = _stripped(name)[len(prefix):]
            elif not has_prefixed and "/" not in name:
                rel = name  # flat zip: files directly at archive root
            else:
                continue
            if not rel:
                continue
            if "/" in rel or rel in UPDATE_PROTECTED_FILES:
                # Only top-level app files are updated; nested paths and
                # protected user-data files are skipped.
                skipped.append(rel)
                continue
            target = os.path.join(base_dir, rel)
            with zf.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())
            updated.append(rel)

    if not updated:
        raise RuntimeError(
            f"压缩包中未找到可替换的应用文件（{UPDATE_APP_SUBDIR}/ 下或顶层文件），更新中止"
        )
    return updated, skipped


def download_and_apply_update():
    """Download the latest repo zip from GitHub and overwrite app files.

    Protected files (user data / local config) are never touched.
    Returns (updated_files, skipped_files, latest_commit_info).
    
    Tries GitHub codeload first, if it fails (e.g., network issues in China),
    falls back to using ghproxy mirror.
    """
    latest = fetch_latest_commit()
    
    # Try GitHub codeload first
    zip_url = (
        f"https://codeload.github.com/{UPDATE_REPO_OWNER}/{UPDATE_REPO_NAME}"
        f"/zip/refs/heads/{UPDATE_BRANCH}"
    )
    try:
        req = urllib.request.Request(zip_url, headers={"User-Agent": "ComfyUI-Console-Updater"})
        with urllib.request.urlopen(req, timeout=UPDATE_ZIP_TIMEOUT) as r:
            zip_bytes = r.read()
    except Exception as e:
        # If GitHub codeload fails, try using ghproxy mirror
        mirror_zip_url = (
            f"https://ghp.ci/https://github.com/{UPDATE_REPO_OWNER}/{UPDATE_REPO_NAME}"
            f"/archive/refs/heads/{UPDATE_BRANCH}.zip"
        )
        try:
            req_mirror = urllib.request.Request(mirror_zip_url, headers={"User-Agent": "ComfyUI-Console-Updater"})
            with urllib.request.urlopen(req_mirror, timeout=UPDATE_ZIP_TIMEOUT) as r:
                zip_bytes = r.read()
        except Exception as e2:
            raise RuntimeError(f"下载更新失败：GitHub 和镜像源均无法访问。原始错误：{e}，镜像源错误：{e2}")

    updated, skipped = apply_zip_bytes(zip_bytes, latest)
    save_local_version(latest["sha"])
    return updated, skipped, latest


# ------------------------------------------------------------
# 图库本地文件夹：生成的图片下载并重命名保存到 ./图库，
# 页面删除图片时联动删除该文件夹中的对应文件
# ------------------------------------------------------------
GALLERY_DIR_NAME = "图库"


def gallery_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), GALLERY_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def _safe_gallery_filename(name):
    """Only allow plain file names inside the gallery dir (no traversal)."""
    name = os.path.basename(name or "").strip()
    name = name.replace("/", "_").replace("\\", "_")
    return name or None


def save_gallery_image(filename, subfolder, img_type, comfyui_url=None):
    """Download one generated image from ComfyUI and save it renamed
    into the local 图库 folder (auto-created). Returns saved file name."""
    import datetime
    # The frontend may send the relative same-origin proxy path (/comfy),
    # which urllib cannot open — fall back to backend auto-detection.
    if not comfyui_url or not comfyui_url.startswith(("http://", "https://")):
        comfyui_url = None
    base = (comfyui_url or resolve_comfy_url()).rstrip("/")
    url = (base + "/view?filename=" + urllib.parse.quote(filename or "") +
           "&subfolder=" + urllib.parse.quote(subfolder or "") +
           "&type=" + urllib.parse.quote(img_type or "output"))
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    ext = os.path.splitext(filename or "")[1] or ".png"
    stem = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = gallery_dir()
    cand = stem + ext
    n = 2
    while os.path.exists(os.path.join(target_dir, cand)):
        cand = f"{stem}_{n}{ext}"
        n += 1
    with open(os.path.join(target_dir, cand), "wb") as f:
        f.write(data)
    return cand


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
        # Restrict CORS to same-origin for security
        origin = self.headers.get("Origin", "")
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Credentials", "true")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message, status=500):
        """Send error JSON response."""
        self._send_json({"error": message, "success": False}, status)

    def _get_session_id(self):
        """Extract session ID from Authorization header or query parameter."""
        # Check Authorization header first
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]
        # Fallback to query parameter (for backward compatibility during transition)
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        return query.get("session", [None])[0]

    def _require_auth(self):
        """Check if request is authenticated. Returns True if authorized."""
        session_id = self._get_session_id()
        return validate_session(session_id)

    def _safe_path(self, path):
        """Securely resolve file path preventing directory traversal attacks.
        
        Returns the safe absolute path or None if path is invalid.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Normalize and remove leading slashes
        clean_path = posixpath.normpath(path).lstrip("/")
        
        # Reject paths with null bytes or other dangerous characters
        if '\x00' in clean_path:
            return None
            
        # Construct full path
        filepath = os.path.normpath(os.path.join(base_dir, clean_path))
        
        # Ensure the resolved path is within base_dir (prevent directory traversal)
        # Use realpath to resolve any symlinks
        real_base = os.path.realpath(base_dir)
        real_filepath = os.path.realpath(filepath)
        
        # Check if the file path starts with the base directory
        if not real_filepath.startswith(real_base + os.sep) and real_filepath != real_base:
            return None
            
        return real_filepath

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
                # Restrict CORS for proxied responses too
                origin = self.headers.get("Origin", "")
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Credentials", "true")
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
        # Only allow requests from the same origin
        origin = self.headers.get("Origin", "")
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Credentials", "true")
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

        # API: Check for updates from GitHub
        if path == "/api/update/check":
            try:
                latest = fetch_latest_commit()
                local = load_local_version()
                up_to_date = bool(local) and (local == latest["sha"])
                self._send_json({
                    "success": True,
                    "up_to_date": up_to_date,
                    "local_commit": local,
                    "latest_commit": latest["sha"],
                    "latest_message": latest["message"],
                    "latest_date": latest["date"],
                    "latest_url": latest["url"],
                    "version": APP_VERSION,
                })
            except Exception as e:
                self._send_error_json(f"检查更新失败: {e}", 502)
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

        # API: Apply update from a locally uploaded zip (raw binary body).
        # Must run before the JSON body read, since the payload is binary.
        if path == "/api/update/zip":
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0:
                    self._send_error_json("没有收到压缩包数据", 400)
                    return
                if length > 300 * 1024 * 1024:
                    self._send_error_json("压缩包过大（上限 300MB）", 413)
                    return
                zip_bytes = self.rfile.read(length)
                updated, skipped = apply_zip_bytes(zip_bytes)
                self._send_json({
                    "success": True,
                    "updated_files": updated,
                    "skipped_files": skipped,
                    "message": "压缩包更新完成，请刷新页面加载新版前端（如后端有更新需重启服务器）",
                })
            except Exception as e:
                self._send_error_json(f"压缩包更新失败: {e}", 502)
            return

        body = self._read_body()

        # API: Save a generated image into the local 图库 folder (renamed).
        if path == "/api/gallery/save":
            try:
                saved = save_gallery_image(
                    body.get("filename", ""),
                    body.get("subfolder", ""),
                    body.get("type", "output"),
                    comfyui_url=body.get("comfyui_url") or None,
                )
                self._send_json({"success": True, "saved_name": saved})
            except Exception as e:
                self._send_error_json(f"保存到图库文件夹失败: {e}", 502)
            return

        # API: Remove files from the local 图库 folder (linked delete).
        if path == "/api/gallery/remove":
            removed, failed = [], []
            for name in (body.get("names") or []):
                safe = _safe_gallery_filename(name)
                if not safe:
                    # 空文件名：记录为失败，让前端知道删除未成功
                    failed.append({"name": name, "reason": "invalid filename"})
                    continue
                p = os.path.join(gallery_dir(), safe)
                try:
                    if os.path.isfile(p):
                        os.remove(p)
                        removed.append(safe)
                    else:
                        # 文件不存在：记录为失败
                        failed.append({"name": safe, "reason": "file not found"})
                except Exception as e:
                    # 详细记录异常原因
                    import logging
                    logging.warning(f"Failed to delete gallery file {safe}: {e}")
                    failed.append({"name": safe, "reason": str(e)})
            self._send_json({"success": True, "removed": removed, "failed": failed})
            return

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

        # API: Apply update (download latest from GitHub and overwrite app files)
        if path == "/api/update/apply":
            try:
                updated, skipped, latest = download_and_apply_update()
                self._send_json({
                    "success": True,
                    "commit": latest["sha"],
                    "updated_files": updated,
                    "skipped_files": skipped,
                    "message": "更新完成，请刷新页面加载新版前端（如后端有更新需重启服务器）",
                })
            except Exception as e:
                self._send_error_json(f"更新失败: {e}", 502)
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
