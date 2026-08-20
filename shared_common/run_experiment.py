import json
import os
import time

from search_space import default_hyp
from summarize_log import read_log, to_text
from train_once import train_once
from llm_optimizer import propose

NOISE = 0.008  # 新資料集(937張)的2σ門檻，原本727張是0.005
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


def run(backend, n_iter=10, seed=0, epochs=100):
    tag = f"llm_{backend}_v2_s{seed}"  # 加上 v2 標記，避免跟舊資料集紀錄混淆
    hist_path = os.path.join(HISTORY_DIR, f"{tag}.json")
    history = load_history(hist_path)

    if history:
        print(f"[resume] 已有 {len(history)} 輪紀錄，接續執行")
        best_map = max(h["map5095"] for h in history)
    else:
        print("[start] 第 0 輪：使用預設超參數建立起點")
        out = train_once(default_hyp(), f"{tag}_i00", seed=seed, epochs=epochs, data_path=DATA_PATH_V2)
        history.append({
            "iteration":      0,
            "backend":        "default",
            "hyp":            out["hyp"],
            "map50":          out["map50"],
            "map5095":        out["map5095"],
            "per_class":      out["per_class"],
            "minutes":        out["minutes"],
            "run_name":       out["run_name"],
            "verdict":        "baseline",
            "diagnosis":      "",
            "hypothesis":     "",
            "changes":        [],
            "base_iteration": None,
            "duplicate_of":   None,
            "notes":          [],
            "parse_error":    None,
        })
        best_map = out["map5095"]
        save_history(hist_path, history)
        print(f"[i00] mAP50-95 = {out['map5095']}  ({out['minutes']} min)")

    while len(history) <= n_iter:
        i = len(history)
        print(f"\n{'='*60}\n[i{i:02d}] 詢問 {backend} ...")

        last_run = history[-1]["run_name"]
        log = read_log(f"C:/YOLO_agent_project/runs/{last_run}/results.csv")
        summary = to_text(log)

        t0 = time.time()
        sug = propose(summary, history, backend, tag=f"{tag}_i{i:02d}")
        llm_sec = round(time.time() - t0, 1)

        if sug["parse_error"]:
            print(f"  ! JSON 解析失敗: {sug['parse_error']}")
        for n in sug["notes"]:
            print(f"  ! {n}")
        if sug["duplicate_of"] is not None:
            print(f"  ! 重複配置，與 iteration {sug['duplicate_of']} 相同")

        print(f"  base : iteration {sug['base_iteration']}")
        for c in sug["changes"]:
            print(f"  變更 : {c.get('param')} {c.get('from')} -> {c.get('to')}")
        print(f"  假設 : {sug['hypothesis'][:100]}")

        out = train_once(sug["hyp"], f"{tag}_i{i:02d}", seed=seed, epochs=epochs, data_path=DATA_PATH_V2)
        verdict = judge(out["map5095"], best_map)

        history.append({
            "iteration":      i,
            "backend":        backend,
            "hyp":            out["hyp"],
            "map50":          out["map50"],
            "map5095":        out["map5095"],
            "per_class":      out["per_class"],
            "minutes":        out["minutes"],
            "llm_seconds":    llm_sec,
            "run_name":       out["run_name"],
            "verdict":        verdict,
            "diagnosis":      sug["diagnosis"],
            "hypothesis":     sug["hypothesis"],
            "changes":        sug["changes"],
            "base_iteration": sug["base_iteration"],
            "duplicate_of":   sug["duplicate_of"],
            "notes":          sug["notes"],
            "parse_error":    sug["parse_error"],
        })

        if out["map5095"] > best_map:
            best_map = out["map5095"]

        save_history(hist_path, history)
        print(f"[i{i:02d}] mAP50-95 = {out['map5095']}  [{verdict}]  "
              f"best = {round(best_map, 4)}")

    print(f"\n{'='*60}\n完成。最佳 mAP50-95 = {round(best_map, 4)}")
    print(f"紀錄已存於 {hist_path}")

    print("\niter | mAP50-95 | verdict   | base | dup")
    for h in history:
        base = h.get("base_iteration")
        dup = h.get("duplicate_of")
        print(f"{h['iteration']:4d} | {h['map5095']:.4f}   | {h['verdict']:9s} | "
              f"{str(base):>4s} | {str(dup):>3s}")


if __name__ == "__main__":
    run(backend="qwen", n_iter=10, seed=0, epochs=100)