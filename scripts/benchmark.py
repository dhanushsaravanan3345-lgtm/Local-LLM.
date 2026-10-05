#!/usr/bin/env python3
"""Measure real performance of the configured model through Ollama (no estimates).

Reports: cold model load time, first-token latency, tokens/s, RAM, GPU/VRAM.
Usage: python scripts/benchmark.py [--model NAME] [--runs 3]
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402

PROMPT = "Explain in about 120 words how a CPU cache works."


def post(path, body, stream=False):
    req = urllib.request.Request(settings.ollama_url + path, data=json.dumps(body).encode(), method="POST")
    return urllib.request.urlopen(req, timeout=600)


def mem_snapshot():
    import psutil
    procs = [p for p in psutil.process_iter(["name", "memory_info"]) if "ollama" in (p.info["name"] or "").lower()]
    rss = sum(p.info["memory_info"].rss for p in procs if p.info["memory_info"]) / 2**30
    vm = psutil.virtual_memory()
    out = {"ollama_rss_gb": round(rss, 2), "system_ram_used_gb": round(vm.used / 2**30, 1)}
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=3).stdout.strip().splitlines()[0].split(",")
        out["gpu_vram_used_mb"], out["gpu_vram_total_mb"], out["gpu_util_percent"] = [float(x) for x in g]
    except Exception:
        pass
    try:
        ps = json.loads(urllib.request.urlopen(settings.ollama_url + "/api/ps", timeout=3).read())["models"]
        if ps:
            out["model_size_gb"] = round(ps[0]["size"] / 2**30, 2)
            out["model_in_vram_gb"] = round(ps[0].get("size_vram", 0) / 2**30, 2)
    except Exception:
        pass
    return out


def one_run(model):
    t0 = time.perf_counter(); first = None; final = {}
    with post("/api/chat", {"model": model, "stream": True, "messages": [{"role": "user", "content": PROMPT}],
                            "options": {"num_ctx": settings.num_ctx}}) as r:
        for line in r:
            o = json.loads(line)
            if first is None and o.get("message", {}).get("content"):
                first = time.perf_counter() - t0
            if o.get("done"):
                final = o
    ed = final.get("eval_duration") or 1
    return {"first_token_s": round(first or 0, 2), "tokens_per_s": round(final.get("eval_count", 0) / (ed / 1e9), 1),
            "load_s": round((final.get("load_duration") or 0) / 1e9, 2), "tokens": final.get("eval_count")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=settings.model)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    try:
        post("/api/generate", {"model": a.model, "keep_alive": 0}).read()  # unload -> measure a cold start
        time.sleep(1)
    except Exception as e:
        sys.exit(f"Cannot talk to Ollama at {settings.ollama_url}: {e}")
    runs = [one_run(a.model)]
    cold = runs[0]
    print(f"Cold start: model load {cold['load_s']}s, first token {cold['first_token_s']}s (incl. load)")
    mem = mem_snapshot()
    for _ in range(a.runs):
        runs.append(one_run(a.model))
    warm = runs[1:]
    result = {"model": a.model, "cold_load_s": cold["load_s"], "cold_first_token_s": cold["first_token_s"],
              "warm_first_token_s_median": statistics.median(r["first_token_s"] for r in warm),
              "warm_tokens_per_s_median": statistics.median(r["tokens_per_s"] for r in warm),
              "memory_while_loaded": mem}
    print(json.dumps(result, indent=2))
    (Path(__file__).resolve().parent.parent / "benchmark_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
