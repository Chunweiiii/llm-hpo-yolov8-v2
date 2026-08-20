from train_once import train_once
import json
 
DATA_PATH_V2 = "C:/YOLO_agent_project/datasets/traffic_light_v2/data.yaml"
SEEDS = [0, 1, 2]
 
# Random Search 在新資料集(937張)六方對照 iter 7 找到的最佳配置
# 來源: history/random_v2_s0.json, iteration 7, 原始單一 seed mAP50-95 = 0.6730
BEST_HYP = {
    "lr0": 0.000521,
    "lrf": 0.085582,
    "momentum": 0.864366,
    "weight_decay": 0.001529,
    "warmup_epochs": 3.04514,
    "box": 5.24837,
    "cls": 0.353769,
    "hsv_v": 0.494374,
    "mosaic": 0.577175,
}
 
 
def mean(xs):
    return sum(xs) / len(xs)
 
 
def stdev(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
 
 
def main():
    results = []
    for seed in SEEDS:
        out = train_once(BEST_HYP, run_name=f"random_v2_verify_s{seed}",
                          seed=seed, epochs=100, data_path=DATA_PATH_V2)
        results.append({"seed": seed, "map50": out["map50"], "map5095": out["map5095"]})
        print(f"[seed={seed}] mAP50-95 = {out['map5095']}")
 
    maps = [r["map5095"] for r in results]
    summary = {
        "method": "Random Search",
        "source": "history/random_v2_s0.json iteration 7",
        "results": results,
        "mean_map5095": round(mean(maps), 4),
        "std_map5095": round(stdev(maps), 4),
        "baseline_map5095": 0.6463,
        "improvement_over_baseline": round(mean(maps) - 0.6463, 4),
    }
 
    with open("random_v2_verify_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
 
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
 
 
if __name__ == "__main__":
    main()