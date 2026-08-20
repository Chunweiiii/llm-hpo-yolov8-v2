import json
import os

from train_once import train_once

OUT_PATH = "C:/YOLO_agent_project/final_verification.json"

BEST_HYP = dict(
    lr0=0.001429,
    lrf=0.01,
    momentum=0.9,
    weight_decay=0.0008,
    warmup_epochs=4.5,
    box=6.0,
    cls=0.9,
    hsv_v=0.6,
    mosaic=1.0,
)

SEEDS = [0, 1, 2]


def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def main():
    results = []
    for seed in SEEDS:
        print(f"\n{'='*50}\n[seed={seed}] 驗證最佳配置 (來源: Sonnet5 i5)")
        out = train_once(BEST_HYP, f"final_best_s{seed}", seed=seed, epochs=100)
        results.append({
            "seed": seed,
            "map50": out["map50"],
            "map5095": out["map5095"],
            "per_class": out["per_class"],
            "minutes": out["minutes"],
        })
        print(f"[seed={seed}] mAP50-95 = {out['map5095']}  ({out['minutes']} min)")

    maps = [r["map5095"] for r in results]
    summary = {
        "source": "Sonnet 5, iteration 5",
        "hyp": BEST_HYP,
        "results": results,
        "mean_map5095": round(mean(maps), 4),
        "std_map5095": round(stdev(maps), 4),
        "baseline_mean": 0.7024,
        "baseline_std": 0.0026,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"最佳配置: mAP50-95 = {summary['mean_map5095']} ± {summary['std_map5095']}")
    print(f"baseline: mAP50-95 = {summary['baseline_mean']} ± {summary['baseline_std']}")
    improvement = summary['mean_map5095'] - summary['baseline_mean']
    print(f"改善幅度: {round(improvement, 4)}")
    print(f"結果已存於 {OUT_PATH}")


if __name__ == "__main__":
    main()