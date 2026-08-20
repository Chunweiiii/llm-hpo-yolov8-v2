import json
import os
import urllib.request

from search_space import validate, to_prompt_text

PROGRAM_PATH = "C:/YOLO_agent_project/prompts/program.md"
TRACE_DIR    = "C:/YOLO_agent_project/traces"

OLLAMA_URL = "http://localhost:11434/api/chat"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

BACKENDS = {
    "qwen":   {"kind": "ollama",    "model": "qwen2.5-coder:7b"},
    "haiku":  {"kind": "anthropic", "model": "claude-haiku-4-5-20251001"},
    "sonnet": {"kind": "anthropic", "model": "claude-sonnet-5"},
    "opus":   {"kind": "anthropic", "model": "claude-opus-5"},
}


# ---------- prompt 組裝 ----------

def load_program():
    with open(PROGRAM_PATH, encoding="utf-8") as f:
        return f.read()


def render_current_state(history):
    if not history:
        return "No previous experiments. This is the first iteration."

    best = max(history, key=lambda h: h["map5095"])
    last = history[-1]

    lines = [
        f"Best configuration so far — iteration {best['iteration']}, "
        f"mAP50-95 = {best['map5095']}:",
        json.dumps(best["hyp"]),
        "",
        f"Most recent configuration — iteration {last['iteration']}, "
        f"mAP50-95 = {last['map5095']} ({last['verdict']}):",
        json.dumps(last["hyp"]),
    ]
    if best["iteration"] != last["iteration"]:
        lines.append("")
        lines.append(
            "Note: the most recent run is not the best one. Consider whether "
            "building on the best configuration is more sensible."
        )
    return "\n".join(lines)


def render_history(history):
    if not history:
        return "No previous experiments."

    best_iter = max(history, key=lambda h: h["map5095"])["iteration"]

    lines = []
    for h in history:
        mark = "  <-- best so far" if h["iteration"] == best_iter else ""
        lines.append(f"--- Iteration {h['iteration']} ---{mark}")
        lines.append(f"hyperparameters: {json.dumps(h['hyp'])}")
        lines.append(f"result mAP50-95: {h['map5095']}  (mAP50: {h['map50']})")
        lines.append(f"per-class: {json.dumps(h['per_class'])}")
        lines.append(f"verdict: {h['verdict']}")
        if h.get("hypothesis"):
            lines.append(f"your previous hypothesis: {h['hypothesis']}")
        lines.append("")
    return "\n".join(lines)


def build_prompt(log_summary, history):
    text = load_program()
    text = text.replace("{{SEARCH_SPACE}}", to_prompt_text())
    text = text.replace("{{CURRENT_STATE}}", render_current_state(history))
    text = text.replace("{{HISTORY}}", render_history(history))
    text = text.replace("{{LOG_SUMMARY}}", log_summary)
    return text


# ---------- 後端呼叫 ----------

def post_json(url, payload, headers):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_ollama(prompt, model):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": 0,
        "options": {"temperature": 0, "num_ctx": 16384},
    }
    out = post_json(OLLAMA_URL, payload, {"Content-Type": "application/json"})
    return out["message"]["content"]


def read_api_key():
    path = "C:/YOLO_agent_project/.env"
    if not os.path.exists(path):
        raise RuntimeError(".env 檔不存在，無法呼叫 Anthropic API")
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(".env 中找不到 ANTHROPIC_API_KEY")


def call_anthropic(prompt, model):
    thinking_models = ("claude-sonnet-5", "claude-opus-5")
    max_tokens = 8000 if model in thinking_models else 2000

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if model not in thinking_models:
        payload["temperature"] = 0

    headers = {
        "Content-Type": "application/json",
        "x-api-key": read_api_key(),
        "anthropic-version": "2023-06-01",
    }
    out = post_json(ANTHROPIC_URL, payload, headers)
    return "".join(block.get("text", "") for block in out["content"])


def ask_llm(prompt, backend):
    if backend not in BACKENDS:
        raise ValueError(f"未知的 backend: {backend}")
    cfg = BACKENDS[backend]
    if cfg["kind"] == "ollama":
        return call_ollama(prompt, cfg["model"])
    return call_anthropic(prompt, cfg["model"])


# ---------- 回應解析 ----------

def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("回應中找不到 JSON 物件")
    return json.loads(text[start : end + 1])


def save_trace(tag, prompt, raw):
    os.makedirs(TRACE_DIR, exist_ok=True)
    with open(os.path.join(TRACE_DIR, f"{tag}_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)
    with open(os.path.join(TRACE_DIR, f"{tag}_response.txt"), "w", encoding="utf-8") as f:
        f.write(raw)


def check_duplicate(hyp, history):
    for h in history:
        if h["hyp"] == hyp:
            return h["iteration"]
    return None


# ---------- 對外主函式 ----------

def propose(log_summary, history, backend, tag="untagged"):
    prompt = build_prompt(log_summary, history)
    raw = ask_llm(prompt, backend)
    save_trace(tag, prompt, raw)

    parse_error = None
    try:
        parsed = extract_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        parse_error = str(e)
        parsed = {}

    hyp, notes = validate(parsed.get("hyperparameters", {}))
    dup_of = check_duplicate(hyp, history)

    return {
        "backend":        backend,
        "hyp":            hyp,
        "diagnosis":      parsed.get("diagnosis", ""),
        "hypothesis":     parsed.get("hypothesis", ""),
        "changes":        parsed.get("changes", []),
        "base_iteration": parsed.get("base_iteration"),
        "duplicate_of":   dup_of,
        "notes":          notes,
        "parse_error":    parse_error,
        "raw":            raw,
    }


if __name__ == "__main__":
    from summarize_log import read_log, to_text

    log = read_log("C:/YOLO_agent_project/runs/baseline_s0/results.csv")
    summary = to_text(log)

    out = propose(summary, history=[], backend="qwen", tag="test")

    print("=" * 60)
    print("parse_error:", out["parse_error"])
    print("base_iteration:", out["base_iteration"])
    print()
    print("diagnosis :", out["diagnosis"])
    print("hypothesis:", out["hypothesis"])
    print()
    for c in out["changes"]:
        print("  ", c)
    print()
    for k, v in out["hyp"].items():
        print(f"   {k:15s} {v}")
    for n in out["notes"]:
        print("  !", n)
    print("=" * 60)