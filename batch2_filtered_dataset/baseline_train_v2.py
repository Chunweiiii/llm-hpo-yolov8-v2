from train_once import train_once
from search_space import default_hyp

import json

DATA_PATH = "C:/YOLO_agent_project/datasets/traffic_light_v2/data.yaml"
SEEDS = [0, 1, 2]


def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def main():
    results = []
    for seed in SEEDS:
        print(f"\n{'='*50}\n[seed={seed}] baseline_v2 訓練開始")
        out = train_once(
            default_hyp(),
            run_name=f"baseline_v2_s{seed}",
            seed=seed,
            epochs=100,
            data_path=DATA_PATH,
        )
        results.append({
            "seed": seed,
            "map50": out["map50"],
            "map5095": out["map5095"],
        })
        print(f"[seed={seed}] mAP50-95 = {out['map5095']}")

    maps = [r["map5095"] for r in results]
    map50s = [r["map50"] for r in results]

    summary = {
        "dataset": "traffic_light_v2 (937 images, filtered+merged)",
        "results": results,
        "mean_map5095": round(mean(maps), 4),
        "std_map5095": round(stdev(maps), 4),
        "mean_map50": round(mean(map50s), 4),
        "std_map50": round(stdev(map50s), 4),
    }

    with open("baseline_v2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"新資料集 baseline: mAP50-95 = {summary['mean_map5095']} \u00b1 {summary['std_map5095']}")
    print(f"對照舊資料集 baseline: mAP50-95 = 0.7024 \u00b1 0.0026")


if __name__ == "__main__":
    main()