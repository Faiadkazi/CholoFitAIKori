import os, sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
ENV = ROOT / "Backend" / ".env"
print("CWD:", ROOT)
print("PYTHON:", sys.executable)
print(".env path:", ENV, "exists:", ENV.exists())

# show first chars only (safe)
if ENV.exists():
    txt = ENV.read_text(encoding="utf-8", errors="ignore")
    print(".env bytes:", len(txt), "preview:", txt[:25].replace("\n","\\n"))

load_dotenv(dotenv_path=ENV)
key = os.getenv("OPENAI_API_KEY")
print("ENV var present:", key is not None, "prefix:", (key[:8] if key else None))

try:
    from openai import OpenAI
    if key:
        client = OpenAI(api_key=key)
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":"ping"}],
            max_tokens=5,
        )
        print("OpenAI OK:", len(r.choices)>0)
    else:
        print("OpenAI SKIP: no key")
except Exception as e:
    print("OpenAI ERROR:", e.__class__.__name__, str(e)[:200])
