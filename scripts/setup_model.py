#!/usr/bin/env python3
"""Inspect this machine, pick the smallest practical model, install/verify Ollama, pull models, write .env.

Usage:
  python scripts/setup_model.py            # inspect + recommend + pull + verify + write .env
  python scripts/setup_model.py --dry-run  # inspect + recommend only
  python scripts/setup_model.py --install-ollama   # also try to install Ollama automatically
  python scripts/setup_model.py --model qwen2.5:1.5b   # force a model
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GB = 2**30


def run(cmd, timeout=5):
    try:
        o = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return o.stdout.strip() if o.returncode == 0 else ""
    except Exception:
        return ""


def find_ollama():
    """PATH is not refreshed in an already-open terminal after install, so also check default locations."""
    p = shutil.which("ollama")
    if p:
        return p
    cands = []
    if sys.platform == "win32":
        cands = [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
                 Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Ollama" / "ollama.exe"]
    elif sys.platform == "darwin":
        cands = [Path("/opt/homebrew/bin/ollama"), Path("/usr/local/bin/ollama"),
                 Path("/Applications/Ollama.app/Contents/Resources/ollama")]
    return next((str(c) for c in cands if c.exists()), None)


def total_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / GB
    except ImportError:
        pass
    if sys.platform == "darwin":
        return int(run(["sysctl", "-n", "hw.memsize"]) or 0) / GB
    if sys.platform.startswith("linux"):
        for l in Path("/proc/meminfo").read_text().splitlines():
            if l.startswith("MemTotal"):
                return int(l.split()[1]) * 1024 / GB
    if sys.platform == "win32":
        out = run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"], 10)
        return int(out or 0) / GB
    return 0.0


def cpu_name():
    if sys.platform == "win32":
        return run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"], 10) or platform.processor()
    if sys.platform == "darwin":
        return run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    for l in Path("/proc/cpuinfo").read_text().splitlines() if Path("/proc/cpuinfo").exists() else []:
        if l.startswith("model name"):
            return l.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def gpu_info():
    out = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if out:
        n, mem = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"kind": "nvidia", "name": n, "vram_gb": float(mem) / 1024}
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return {"kind": "apple", "name": "Apple Silicon (unified memory)", "vram_gb": None}
    if shutil.which("rocm-smi"):
        return {"kind": "amd", "name": "AMD GPU (ROCm)", "vram_gb": None}
    return {"kind": "none", "name": "No supported discrete GPU detected", "vram_gb": None}


def inspect():
    du = shutil.disk_usage(ROOT)
    return {"os": f"{platform.system()} {platform.release()} ({platform.machine()})", "cpu": cpu_name(),
            "cores": os.cpu_count(), "ram_gb": round(total_ram_gb(), 1), "gpu": gpu_info(),
            "disk_free_gb": round(du.free / GB, 1), "python": platform.python_version()}


def recommend(hw):
    """Smallest model that is comfortable for the hardware (4-bit quantized builds served by Ollama)."""
    ram, gpu = hw["ram_gb"], hw["gpu"]
    if gpu["kind"] == "nvidia" and gpu["vram_gb"] >= 6 and ram >= 12:
        model, why = "qwen2.5:7b", f"{gpu['vram_gb']:.0f} GB VRAM fits a 4-bit 7B (~4.7 GB) fully on the GPU"
    elif gpu["kind"] == "nvidia" and gpu["vram_gb"] >= 3:
        model, why = "qwen2.5:3b", f"{gpu['vram_gb']:.0f} GB VRAM fits a 4-bit 3B (~1.9 GB)"
    elif gpu["kind"] == "apple" and ram >= 16:
        model, why = "qwen2.5:7b", "Apple Silicon with >=16 GB unified memory runs a 4-bit 7B well"
    elif ram >= 6:
        model, why = "qwen2.5:3b", "CPU / shared-memory inference: 3B (~1.9 GB) is the best speed/quality balance"
    else:
        model, why = "qwen2.5:1.5b", "low RAM: 1.5B (~1 GB) keeps the machine responsive"
    embed = "all-minilm" if ram < 6 else "nomic-embed-text"
    return model, embed, why


def ollama_up(url="http://127.0.0.1:11434"):
    try:
        urllib.request.urlopen(url + "/api/version", timeout=2)
        return True
    except Exception:
        return False


def install_ollama():
    if sys.platform == "win32" and shutil.which("winget"):
        return subprocess.call(["winget", "install", "-e", "--id", "Ollama.Ollama"]) == 0
    if sys.platform == "darwin" and shutil.which("brew"):
        return subprocess.call(["brew", "install", "ollama"]) == 0
    if sys.platform.startswith("linux"):
        return subprocess.call("curl -fsSL https://ollama.com/install.sh | sh", shell=True) == 0
    return False


def ensure_server(exe):
    for _ in range(10):  # the Windows/macOS installer usually auto-starts Ollama; give it a moment
        if ollama_up():
            return True
        time.sleep(1)
    print("Starting `ollama serve` in the background...")
    kw = {"creationflags": 0x08000000} if sys.platform == "win32" else {"start_new_session": True}
    subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
    for _ in range(30):
        if ollama_up():
            return True
        time.sleep(1)
    return False


def write_env(model, embed):
    src = (ROOT / ".env.example").read_text()
    lines = []
    for l in src.splitlines():
        if l.startswith("OLLAMA_MODEL="):
            l = f"OLLAMA_MODEL={model}"
        elif l.startswith("EMBED_MODEL="):
            l = f"EMBED_MODEL={embed}"
        lines.append(l)
    (ROOT / ".env").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--install-ollama", action="store_true")
    ap.add_argument("--model")
    a_ = ap.parse_args()

    hw = inspect()
    print("== Hardware ==")
    for k, v in hw.items():
        print(f"  {k:13}: {v}")
    model, embed, why = recommend(hw)
    model = a_.model or model
    print(f"\n== Recommendation ==\n  chat model     : {model}\n  embedding model: {embed}\n  reason         : {why}")
    if hw["disk_free_gb"] < 8:
        print("  WARNING: less than 8 GB free disk space.")
    (ROOT / "hardware_report.json").write_text(json.dumps(hw, indent=2, default=str))
    if a_.dry_run:
        return 0

    exe = find_ollama()
    if not exe:
        if a_.install_ollama and install_ollama():
            exe = find_ollama()
        if not exe:
            print("\nOllama is not installed or not found. Install it from https://ollama.com/download "
                  "(or use --install-ollama), then OPEN A NEW TERMINAL and re-run this script.")
            return 1
    print(f"Using Ollama at: {exe}")
    if not ensure_server(exe):
        print("Could not reach Ollama on 127.0.0.1:11434. Start the Ollama app (or run `ollama serve`) and re-run.")
        return 1
    for m in (model, embed):
        print(f"\nPulling {m} (needs internet once; afterwards everything is offline)...")
        if subprocess.call([exe, "pull", m]) != 0:
            print(f"Pull failed for {m}. Check your internet connection and retry.")
            return 1
    write_env(model, embed)
    print("\nWrote .env. Verifying the model answers...")
    t = time.time()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", method="POST",
                                 data=json.dumps({"model": model, "prompt": "Reply with the single word: ready", "stream": False}).encode())
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=300).read())
        print(f"  model said: {r.get('response', '').strip()!r}  ({time.time() - t:.1f}s incl. load)")
    except Exception as e:
        print(f"  verification failed: {e}")
        return 1
    print("\nDone. Start the app:  python -m uvicorn app.main:app --host 127.0.0.1 --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())