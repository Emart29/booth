"""
Does BOOTH's self-reported confidence predict correct answers?

Runs booth.check() over a fixed 300-question sample of TriviaQA (rc.nocontext,
validation split) for each model, grades every answer against TriviaQA's
human-curated answer aliases, and reports how well attempt-0 confidence
separates right answers from wrong ones: accuracy by confidence bucket, how
many wrong answers still cleared the 0.7 threshold, a threshold sweep, AUROC,
Brier score, ECE, and what the single default retry did.

Models:
- Ollama (local, no key): qwen2.5:7b-instruct and llama3.2:3b by default.
- Groq (optional): llama-3.3-70b-versatile, only if GROQ_API_KEY is set.

Usage:
    python run_example.py                    # run (resumable) + print report
    python run_example.py --limit 20         # quick smoke test
    python run_example.py --report-only      # re-print report from raw outputs
    python run_example.py --audit 50         # print the grader-audit sample
    python run_example.py --build-sample validation.parquet
        # rebuild triviaqa_sample.json from the TriviaQA parquet (needs pyarrow)

Requirements:
- pip install boothpy==0.5.6
- A running Ollama server (`ollama serve`) with the models pulled:
  `ollama pull qwen2.5:7b-instruct` and `ollama pull llama3.2:3b`
- Nothing else: model calls use urllib from the standard library.
- Optional: GROQ_API_KEY in the environment for the hosted model.
"""

import argparse
import gzip
import json
import math
import os
import platform
import random
import re
import string
import sys
import time
import urllib.error
import urllib.request

try:
    import booth
except ImportError:
    sys.exit("BOOTH is not installed. Run: pip install boothpy==0.5.6")

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(HERE, "triviaqa_sample.json")
RAW_PATH = os.path.join(HERE, "raw_outputs.jsonl")

SAMPLE_SIZE = 300
SAMPLE_SEED = 0
THRESHOLD = 0.7  # BOOTH's default, passed explicitly so the run doesn't depend on it
MAX_RETRIES = 1  # BOOTH's default
TEMPERATURE = 0.0
SEED = 0

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODELS = ["qwen2.5:7b-instruct", "llama3.2:3b"]
GROQ_MODEL = "llama-3.3-70b-versatile"

BUCKETS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0), (1.0, 1.0)]


# ---------------------------------------------------------------- dataset

def build_sample(parquet_path):
    """Sample SAMPLE_SIZE questions from TriviaQA rc.nocontext validation."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("--build-sample needs pyarrow (example-only): pip install pyarrow")
    table = pq.read_table(parquet_path, columns=["question_id", "question", "answer"]).to_pylist()
    # rc.nocontext lists many questions twice (once per evidence domain), so
    # sample unique question_ids, not rows.
    rows = sorted({r["question_id"]: r for r in table}.values(), key=lambda r: r["question_id"])
    picked = random.Random(SAMPLE_SEED).sample(rows, SAMPLE_SIZE)
    sample = [
        {
            "id": r["question_id"],
            "question": r["question"],
            "value": r["answer"]["value"],
            "aliases": sorted(set(r["answer"]["aliases"]) | {r["answer"]["value"]}),
        }
        for r in picked
    ]
    with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(sample)} questions (of {len(rows)} unique, {len(table)} rows) to {SAMPLE_PATH}")


def load_sample():
    if not os.path.exists(SAMPLE_PATH):
        sys.exit(f"{SAMPLE_PATH} is missing. Rebuild it with --build-sample.")
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- grading

def normalize(text):
    """SQuAD-style: lowercase, drop punctuation and articles, collapse spaces."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def grade(answer, aliases):
    """Correct if the normalized answer equals an alias or contains one as whole words."""
    if not answer:
        return False
    a = normalize(answer)
    if not a:
        return False
    for alias in aliases:
        n = normalize(alias)
        if n and (a == n or f" {n} " in f" {a} "):
            return True
    return False


# ---------------------------------------------------------------- model calls

def _post_json(url, payload, headers=None, timeout=600):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _get_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def ollama_call_fn(model):
    def call(prompt):
        r = _post_json(
            f"{OLLAMA_URL}/api/generate",
            {"model": model, "prompt": prompt, "stream": False,
             "options": {"temperature": TEMPERATURE, "seed": SEED}},
        )
        return r["response"]
    return call


def groq_call_fn(model, key):
    def call(prompt):
        r = _post_json(
            "https://api.groq.com/openai/v1/chat/completions",
            {"model": model, "temperature": TEMPERATURE, "seed": SEED,
             "messages": [{"role": "user", "content": prompt}]},
            headers={"Authorization": f"Bearer {key}"},
            timeout=120,
        )
        return r["choices"][0]["message"]["content"]
    return call


def ollama_info(models):
    try:
        version = _get_json(f"{OLLAMA_URL}/api/version")["version"]
        tags = {m["name"]: m["digest"] for m in _get_json(f"{OLLAMA_URL}/api/tags")["models"]}
    except (urllib.error.URLError, OSError):
        sys.exit(f"Is Ollama running at {OLLAMA_URL}? Start it with `ollama serve`.")
    missing = [m for m in models if m not in tags]
    if missing:
        sys.exit("Model(s) not pulled: " + ", ".join(missing)
                 + ". Run: " + " && ".join(f"ollama pull {m}" for m in missing))
    return version, {m: tags[m] for m in models}


# ---------------------------------------------------------------- running

def load_raw():
    rows = []
    if os.path.exists(RAW_PATH):
        with open(RAW_PATH, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    elif os.path.exists(RAW_PATH + ".gz"):
        with gzip.open(RAW_PATH + ".gz", "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return rows


def run_model(name, call_fn, sample, done):
    todo = [q for q in sample if (name, q["id"]) not in done]
    print(f"[{name}] {len(sample) - len(todo)} already done, {len(todo)} to run", flush=True)
    start = time.time()
    with open(RAW_PATH, "a", encoding="utf-8") as out:
        for i, q in enumerate(todo, 1):
            t = time.time()
            result = booth.check(call_fn, q["question"], threshold=THRESHOLD, max_retries=MAX_RETRIES)
            row = {
                "model": name,
                "id": q["id"],
                "question": q["question"],
                "aliases": q["aliases"],
                "seconds": round(time.time() - t, 2),
                "result": result.to_dict(),
                "attempt_correct": [grade(a.answer, q["aliases"]) for a in result.attempts],
                "final_correct": grade(result.answer, q["aliases"]),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"[{name}] {i}/{len(todo)} ({time.time() - start:.0f}s)", flush=True)


# ---------------------------------------------------------------- statistics

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def auroc(pairs):
    """P(conf of a random correct > conf of a random wrong), ties count 1/2."""
    pos = [c for c, y in pairs if y]
    neg = [c for c, y in pairs if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def brier(pairs):
    return sum((c - y) ** 2 for c, y in pairs) / len(pairs)


def ece(pairs, bins=10):
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inb = [(c, y) for c, y in pairs if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if inb:
            gap = abs(sum(c for c, _ in inb) / len(inb) - sum(y for _, y in inb) / len(inb))
            total += len(inb) / len(pairs) * gap
    return total


def in_bucket(c, lo, hi):
    return c == 1.0 if lo == hi else lo <= c < hi


def pct(k, n):
    return f"{100 * k / n:.1f}%" if n else "n/a"


def fmt_ci(k, n):
    lo, hi = wilson(k, n)
    return f"{pct(k, n)} [{100 * lo:.0f}–{100 * hi:.0f}%]" if n else "n/a"


# ---------------------------------------------------------------- report

def report(rows, model):
    rows = [r for r in rows if r["model"] == model]
    if not rows:
        return
    n = len(rows)
    a0 = [r["result"]["attempts"][0] for r in rows]
    excluded = {
        "call_failed": sum(a["call_failed"] for a in a0),
        "parse_failure": sum((not a["parse_ok"]) and not a["call_failed"] for a in a0),
        "ambiguous": sum(a["parse_ok"] and a["ambiguous"] for a in a0),
    }
    usable = [(a["confidence"], r["attempt_correct"][0]) for r, a in zip(rows, a0)
              if a["parse_ok"] and not a["ambiguous"] and a["confidence"] is not None]
    m = len(usable)
    correct = sum(y for _, y in usable)

    print(f"\n## {model}\n")
    print(f"Questions: {n}. Usable attempt-0 answers (parsed, not ambiguous, call ok): {m}.")
    print("Excluded from calibration: " + ", ".join(f"{k}={v}" for k, v in excluded.items()) + ".")
    print(f"Attempt-0 accuracy on usable answers: {correct}/{m} = {fmt_ci(correct, m)}.")
    secs = [r["seconds"] for r in rows]
    print(f"Wall time per question (incl. retries): median {sorted(secs)[len(secs) // 2]:.1f}s.")

    print("\n### Accuracy by attempt-0 confidence bucket\n")
    print("| Confidence | n | Correct | Accuracy [95% Wilson] |")
    print("|---|---:|---:|---|")
    for lo, hi in BUCKETS:
        inb = [y for c, y in usable if in_bucket(c, lo, hi)]
        label = "= 1.0" if lo == hi else f"[{lo:.1f}, {hi:.1f})"
        print(f"| {label} | {len(inb)} | {sum(inb)} | {fmt_ci(sum(inb), len(inb))} |")

    values = {}
    for c, _ in usable:
        values[c] = values.get(c, 0) + 1
    print("\nDistinct confidence values emitted: "
          + ", ".join(f"{c:g}×{k}" for c, k in sorted(values.items())))

    above = [y for c, y in usable if c >= THRESHOLD]
    below = [y for c, y in usable if c < THRESHOLD]
    wrong = m - correct
    wrong_above = len(above) - sum(above)
    print(f"\n### The {THRESHOLD} gate on attempt 0\n")
    print("| | Value |")
    print("|---|---|")
    print(f"| Accuracy, confidence ≥ {THRESHOLD} | {sum(above)}/{len(above)} = {fmt_ci(sum(above), len(above))} |")
    print(f"| Accuracy, confidence < {THRESHOLD} | {sum(below)}/{len(below)} = {fmt_ci(sum(below), len(below))} |")
    print(f"| Wrong answers that cleared {THRESHOLD} | {wrong_above} |")
    print(f"| ...as share of all wrong answers (missed by gate) | {wrong_above}/{wrong} = {pct(wrong_above, wrong)} |")
    print(f"| ...as share of all answers clearing {THRESHOLD} (false-accept rate) | {wrong_above}/{len(above)} = {pct(wrong_above, len(above))} |")
    print(f"| Correct answers held back (< {THRESHOLD}) | {sum(below)}/{correct} = {pct(sum(below), correct)} of correct |")

    print("\n### Threshold sweep (attempt 0 only)\n")
    print("| Threshold | Coverage | Accuracy of accepted | Wrong accepted |")
    print("|---:|---:|---:|---:|")
    for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0):
        acc = [y for c, y in usable if c >= t]
        print(f"| {t:.2f} | {len(acc)}/{m} = {pct(len(acc), m)} | {pct(sum(acc), len(acc))} | {len(acc) - sum(acc)} |")

    if m:
        print("\n### Threshold-free summaries (attempt 0)\n")
        print(f"- AUROC (0.5 = confidence doesn't rank correct above wrong): {auroc(usable):.3f}")
        print(f"- Brier score (lower is better; always predicting base rate gives "
              f"{(correct / m) * (1 - correct / m):.3f}): {brier(usable):.3f}")
        print(f"- ECE, 10 equal-width bins: {ece(usable):.3f}")

    print(f"\n### What the default retry (max_retries={MAX_RETRIES}) did\n")
    retried = [r for r in rows if len(r["result"]["attempts"]) > 1]
    statuses = {}
    for r in rows:
        statuses[r["result"]["status"]] = statuses.get(r["result"]["status"], 0) + 1
    print("Final statuses: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())))
    repaired = [r for r in retried if r["result"]["status"] == "REPAIRED"]
    pairs = [r["attempt_correct"][:2] for r in retried if len(r["attempt_correct"]) > 1]
    print(f"- Retried: {len(retried)}. Came back REPAIRED: {len(repaired)}, "
          f"of which correct: {sum(r['final_correct'] for r in repaired)}.")
    print(f"- Attempt 0 → attempt 1: wrong→right {sum((not x) and y for x, y in pairs)}, "
          f"right→wrong {sum(x and (not y) for x, y in pairs)}, "
          f"right→right {sum(x and y for x, y in pairs)}, wrong→wrong {sum((not x) and (not y) for x, y in pairs)}.")
    ok = [r for r in rows if r["result"]["ok"]]
    ok_correct = sum(r["final_correct"] for r in ok)
    print(f"- End to end, result.ok answers: {len(ok)}/{n}; correct {ok_correct}/{len(ok)} = "
          f"{pct(ok_correct, len(ok))}; wrong but ok: {len(ok) - ok_correct}.")


def print_audit(rows, k):
    """Fixed random sample per model for a manual check of the grader."""
    for model in sorted({r["model"] for r in rows}):
        mrows = sorted((r for r in rows if r["model"] == model), key=lambda r: r["id"])
        pick = random.Random(1).sample(mrows, min(k, len(mrows)))
        print(f"\n## Grader audit sample: {model}\n")
        print("| id | question | attempt-0 answer | aliases (first 4) | grader |")
        print("|---|---|---|---|---|")
        for r in sorted(pick, key=lambda r: r["id"]):
            a = r["result"]["attempts"][0]["answer"]
            q = r["question"].replace("|", "/")
            al = "; ".join(r["aliases"][:4]).replace("|", "/")
            print(f"| {r['id']} | {q} | {str(a).replace('|', '/')} | {al} | "
                  f"{'correct' if r['attempt_correct'][0] else 'wrong'} |")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--models", default=",".join(OLLAMA_MODELS), help="comma-separated Ollama models")
    p.add_argument("--limit", type=int, default=None, help="only the first N sampled questions")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--audit", type=int, default=0, metavar="K")
    p.add_argument("--build-sample", metavar="PARQUET")
    args = p.parse_args()

    if args.build_sample:
        build_sample(args.build_sample)
        return

    sample = load_sample()[: args.limit]
    models = [m for m in args.models.split(",") if m]
    groq_key = os.environ.get("GROQ_API_KEY")

    config = {
        "booth": getattr(booth, "__version__", "unknown"),
        "python": platform.python_version(),
        "system": f"{platform.system()} {platform.machine()}",
        "questions": len(sample),
        "threshold": THRESHOLD,
        "max_retries": MAX_RETRIES,
        "temperature": TEMPERATURE,
        "seed": SEED,
    }
    if not args.report_only:
        version, digests = ollama_info(models)
        config.update({"ollama": version, "models": digests})
        config["groq"] = GROQ_MODEL if groq_key else "skipped (GROQ_API_KEY not set)"
    print("CONFIG " + json.dumps(config, sort_keys=True))

    if not args.report_only:
        done = {(r["model"], r["id"]) for r in load_raw()}
        for model in models:
            run_model(model, ollama_call_fn(model), sample, done)
        if groq_key:
            run_model(f"groq/{GROQ_MODEL}", groq_call_fn(GROQ_MODEL, groq_key), sample, done)
        else:
            print("Skipping Groq: set GROQ_API_KEY to include the hosted model.")

    ids = {q["id"] for q in sample}
    rows = [r for r in load_raw() if r["id"] in ids]
    if not rows:
        sys.exit("No raw outputs yet. Run without --report-only first.")
    for model in [m for m in models] + [f"groq/{GROQ_MODEL}"]:
        report(rows, model)
    if args.audit:
        print_audit(rows, args.audit)


if __name__ == "__main__":
    main()
