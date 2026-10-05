"""Live smoke test: starts a real uvicorn server (against the test fake Ollama), exercises it over HTTP, stops it."""
import os, shutil, subprocess, sys, tempfile, time
import httpx

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = tempfile.mkdtemp()
env = {**os.environ, "DATA_DIR": data, "OLLAMA_MODEL": "fake-model:1b", "OLLAMA_URL": "http://127.0.0.1:11999"}
fake = subprocess.Popen([sys.executable, "-m", "uvicorn", "fake_ollama:app", "--port", "11999", "--log-level", "error"],
                        cwd=R + "/tests", env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
app = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000", "--log-level", "warning"],
                       cwd=R, env=env, stdout=open(data + "/app.log", "w"), stderr=subprocess.STDOUT)
b = "http://127.0.0.1:8000"
try:
    for _ in range(60):
        try:
            httpx.get(b + "/api/health", timeout=1); break
        except Exception:
            time.sleep(0.25)
    print("health   :", httpx.get(b + "/api/health").json())
    print("static   :", "My Local AI" in httpx.get(b + "/").text, httpx.get(b + "/app.js").status_code, httpx.get(b + "/style.css").status_code)
    with httpx.stream("POST", b + "/api/chat/stream", json={"message": "hello live server"}) as r:
        evs = [l for l in r.iter_lines() if l.startswith("data:")]
    print("stream   :", len(evs), "events | last:", evs[-1][:100])
    with httpx.stream("POST", b + "/api/chat/stream", json={"message": "word " * 10}) as r:  # Stop = disconnect mid-stream
        n = 0
        for l in r.iter_lines():
            if l.startswith("data:"):
                n += 1
                if n == 4:
                    break
    time.sleep(1)
    cid = httpx.get(b + "/api/conversations").json()["conversations"][0]["id"]
    msgs = httpx.get(f"{b}/api/conversations/{cid}").json()["messages"]
    print("abort    : alive =", httpx.get(b + "/api/health").status_code == 200, "| roles =", [m["role"] for m in msgs], "| partial =", repr(msgs[-1]["content"][:30]))
    print("bad json :", httpx.post(b + "/api/chat", content=b"{bad", headers={"content-type": "application/json"}).json())
    print("404      :", httpx.get(b + "/api/nope").json())
finally:
    app.terminate(); fake.terminate(); time.sleep(0.5)
    log = open(data + "/app.log").read()
    print("server log:", log[-300:].strip() or "(clean, no errors)")
    shutil.rmtree(data, ignore_errors=True)
