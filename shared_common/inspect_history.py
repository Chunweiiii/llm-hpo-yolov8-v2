import json

PATH = "C:/YOLO_agent_project/history/llm_qwen_s0.json"

with open(PATH, encoding="utf-8") as f:
    hist = json.load(f)

keys = ["lr0", "lrf", "momentum", "weight_decay",
        "warmup_epochs", "box", "cls", "hsv_v", "mosaic"]

print("iter | mAP50-95 | verdict   | " + " | ".join(f"{k[:7]:>7s}" for k in keys))
print("-" * 110)
for h in hist:
    vals = " | ".join(f"{h['hyp'][k]:7.4f}" for k in keys)
    print(f"{h['iteration']:4d} | {h['map5095']:.4f}   | {h['verdict']:9s} | {vals}")

print("\n" + "=" * 60)
for h in hist:
    if h["iteration"] == 0:
        continue
    chg = ", ".join(
        f"{c.get('param')}: {c.get('from')}->{c.get('to')}" for c in h["changes"]
    ) or "(無變更)"
    print(f"\n[i{h['iteration']:02d}] {h['verdict']}  mAP={h['map5095']}")
    print(f"  changes  : {chg}")
    print(f"  hypothesis: {h['hypothesis'][:150]}")
    if h["notes"]:
        print(f"  notes    : {h['notes']}")
    if h["parse_error"]:
        print(f"  ! parse_error: {h['parse_error']}")