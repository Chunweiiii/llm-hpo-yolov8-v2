import json
import matplotlib.pyplot as plt

METHODS = {
    "Random":       "history/random_v2_s0.json",
    "Optuna":       "history/optuna_v2_s0.json",
    "Qwen 2.5-7B":  "history/llm_qwen_v2_s0.json",
    "Haiku 4.5":    "history/llm_haiku_v2_s0.json",
    "Sonnet 5":     "history/llm_sonnet_v2_s0.json",
    "Opus 5":       "history/llm_opus_v2_s0.json",
}

COLORS = {
    "Random":       "#B4B2A9",
    "Optuna":       "#5F5E5A",
    "Qwen 2.5-7B":  "#D85A30",
    "Haiku 4.5":    "#85B7EB",
    "Sonnet 5":     "#378ADD",
    "Opus 5":       "#185FA5",
}

LINESTYLES = {
    "Random":       "--",
    "Optuna":       "--",
    "Qwen 2.5-7B":  "-",
    "Haiku 4.5":    "-",
    "Sonnet 5":     "-",
    "Opus 5":       "-",
}


def best_so_far(history):
    iters, best = [], []
    running_max = -1
    for h in history:
        running_max = max(running_max, h["map5095"])
        iters.append(h["iteration"])
        best.append(running_max)
    return iters, best


def main():
    plt.figure(figsize=(8, 5.5))

    for name, path in METHODS.items():
        with open(path, encoding="utf-8") as f:
            history = json.load(f)
        iters, best = best_so_far(history)
        plt.plot(
            iters, best,
            label=name,
            color=COLORS[name],
            linestyle=LINESTYLES[name],
            linewidth=2,
            marker="o",
            markersize=4,
        )

    plt.axhline(
        y=0.6463,
        color="#5F5E5A",
        linestyle=":",
        linewidth=1,
        alpha=0.6,
        label="Default baseline (3-seed)",
    )

    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far mAP@0.5:0.95")
    plt.title("Convergence Comparison — Filtered & Augmented Dataset (937 images)")
    plt.legend(loc="upper left", fontsize=9, framealpha=0.95, ncol=1)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.3, 10.3)
    plt.tight_layout()

    plt.savefig("convergence_comparison_v2.png", dpi=300, bbox_inches="tight")
    print("已存檔: convergence_comparison_v2.png")


if __name__ == "__main__":
    main()