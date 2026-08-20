import json
import os
import shutil

OUT_DIR = "runs_for_handoff"

TAGS = {
    "random":  "history/random_s0.json",
    "optuna":  "history/optuna_s0.json",
    "qwen":    "history/llm_qwen_s0.json",
    "haiku":   "history/llm_haiku_s0.json",
    "sonnet":  "history/llm_sonnet_s0.json",
    "opus":    "history/llm_opus_s0.json",
}


def copy_run(run_name):
    src = os.path.join("runs", run_name)
    dst = os.path.join(OUT_DIR, run_name)
    if not os.path.exists(src):
        print(f"  ! 找不到 {src}，跳過")
        return
    if os.path.exists(dst):
        return
    shutil.copytree(src, dst)
    print(f"  已複製: {run_name}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for tag, path in TAGS.items():
        print(f"\n=== {tag} ===")
        with open(path, encoding="utf-8") as f:
            history = json.load(f)

        best = max(history, key=lambda h: h["map5095"])

        copy_run(history[0]["run_name"])   # baseline (iteration 0)
        copy_run(best["run_name"])          # 最佳輪次

    for name in ["baseline_s0", "baseline_s1", "baseline_s2",
                 "final_best_s0", "final_best_s1", "final_best_s2"]:
        copy_run(name)

    print(f"\n完成，精簡版存於 {OUT_DIR}/")


if __name__ == "__main__":
    main()