import random

# ===== 凍結參數（不參與搜尋，但明確傳入以確保可重現）=====
FROZEN = dict(
    optimizer="AdamW",
    hsv_h=0.015,
    flipud=0.0,
)

# ===== 搜尋空間：9 個可調參數 =====
SPACE = {
    "lr0":           {"low": 0.0002, "high": 0.005,  "default": 0.001429},
    "lrf":           {"low": 0.005,  "high": 0.2,    "default": 0.01},
    "momentum":      {"low": 0.85,   "high": 0.98,   "default": 0.9},
    "weight_decay":  {"low": 0.0001, "high": 0.002,  "default": 0.0005},
    "warmup_epochs": {"low": 1.0,    "high": 5.0,    "default": 3.0},
    "box":           {"low": 5.0,    "high": 10.0,   "default": 7.5},
    "cls":           {"low": 0.3,    "high": 1.5,    "default": 0.5},
    "hsv_v":         {"low": 0.2,    "high": 0.6,    "default": 0.4},
    "mosaic":        {"low": 0.5,    "high": 1.0,    "default": 1.0},
}


def default_hyp():
    return {name: spec["default"] for name, spec in SPACE.items()}


def sample_random(rng):
    return {
        name: round(rng.uniform(spec["low"], spec["high"]), 6)
        for name, spec in SPACE.items()
    }


def validate(raw):
    clean, notes = {}, []

    for name in raw:
        if name not in SPACE:
            notes.append(f"忽略未知參數: {name}")

    for name, spec in SPACE.items():
        if name not in raw:
            clean[name] = spec["default"]
            notes.append(f"缺少 {name}，改用預設值 {spec['default']}")
            continue

        try:
            v = float(raw[name])
        except (TypeError, ValueError):
            clean[name] = spec["default"]
            notes.append(f"{name} 非數值，改用預設值 {spec['default']}")
            continue

        if v < spec["low"]:
            notes.append(f"{name}={v} 低於下限，修正為 {spec['low']}")
            v = spec["low"]
        elif v > spec["high"]:
            notes.append(f"{name}={v} 超過上限，修正為 {spec['high']}")
            v = spec["high"]

        clean[name] = round(v, 6)

    return clean, notes


def to_prompt_text():
    lines = []
    for name, spec in SPACE.items():
        lines.append(
            f"- {name}: range [{spec['low']}, {spec['high']}], default {spec['default']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print("--- default ---")
    print(default_hyp())

    print("\n--- random (seed=42) ---")
    print(sample_random(random.Random(42)))

    print("\n--- validate 測試 ---")
    bad = {"lr0": 99, "momentum": "abc", "cls": 0.8, "imgsz": 1280}
    clean, notes = validate(bad)
    print("修正後:", clean)
    for n in notes:
        print("  !", n)

    print("\n--- prompt 文字 ---")
    print(to_prompt_text())