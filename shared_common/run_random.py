import json
import os
import random

from search_space import default_hyp, sample_random
from train_once import train_once

NOISE = 0.008  # 新資料集(937張)的2σ門檻
HISTORY_DIR = "C:/YOLO_agent_project/history"
DATA_PATH_V2 = "C:/YOLO_agent_project/datasets/traffic_light_v2/data.yaml"


def load_history(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(path, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def judge(new_map, best_map):
    if new_map > best_map + NOISE:
        return "improved"
    if new_map < best_map - NOISE:
        return "regressed"
    return "tie"


def run(n_iter=10, seed=0, epochs=100, rng_seed=42):
    tag = f"random_v2_s{seed}"  # 加上 v2 標記
    hist_path = os.path.join(HISTORY_DIR, f"{tag}.json")
    history = load_history(hist_path)

    if history:
        print(f"[resume] 已有 {len(history)} 輪紀錄，接續執行")
        best_map = max(h["map5095"] for h in history)
    else:
        print("[start] 第 0 輪：使用預設超參數建立起點")
        out = train_once(default_hyp(), f"{tag}_i00", seed=seed, epochs=epochs, data_path=DATA_PATH_V2)
        history.append({
            "iteration": 0,
            "method":    "default",
            "hyp":       out["hyp"],
            "map50":     out["map50"],
            "map5095":   out["map5095"],
            "per_class": out["per_class"],
            "minutes":   out["minutes"],
            "run_name":  out["run_name"],
            "verdict":   "baseline",
        })
        best_map = out["map5095"]
        save_history(hist_path, history)
        print(f"[i00] mAP50-95 = {out['map5095']}  ({out['minutes']} min)")

    while len(history) <= n_iter:
        i = len(history)
        print(f"\n{'='*60}\n[i{i:02d}] random sampling ...")

        rng = random.Random(rng_seed + i)
        hyp = sample_random(rng)

        out = train_once(hyp, f"{tag}_i{i:02d}", seed=seed, epochs=epochs, data_path=DATA_PATH_V2)
        verdict = judge(out["map5095"], best_map)

        history.append({
            "iteration": i,
            "method":    "random",
            "hyp":       out["hyp"],
            "map50":     out["map50"],
            "map5095":   out["map5095"],
            "per_class": out["per_class"],
            "minutes":   out["minutes"],
            "run_name":  out["run_name"],
            "verdict":   verdict,
        })

        if out["map5095"] > best_map:
            best_map = out["map5095"]

        save_history(hist_path, history)
        print(f"[i{i:02d}] mAP50-95 = {out['map5095']}  [{verdict}]  "
              f"best = {round(best_map, 4)}")

    print(f"\n{'='*60}\n完成。最佳 mAP50-95 = {round(best_map, 4)}")
    print(f"紀錄已存於 {hist_path}")

    print("\niter | mAP50-95 | verdict")
    for h in history:
        print(f"{h['iteration']:4d} | {h['map5095']:.4f}   | {h['verdict']}")


if __name__ == "__main__":
    run(n_iter=10, seed=0, epochs=100)