import csv

NOISE = 0.005    # 2σ threshold measured from 3-seed baseline
NEAR = 0.02      # practical "good enough" threshold


def read_log(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [n.strip() for n in reader.fieldnames]
        return [{k: float(v) for k, v in row.items()} for row in reader]


def mean(xs):
    return sum(xs) / len(xs)


def smooth(values, w=5):
    out = []
    for i in range(len(values)):
        lo = max(0, i - w // 2)
        hi = min(len(values), i + w // 2 + 1)
        out.append(mean(values[lo:hi]))
    return out


def trend(values, rel_tol=0.02):
    n = len(values)
    if n < 4:
        return "insufficient_data"
    first, second = mean(values[: n // 2]), mean(values[n // 2 :])
    if first == 0:
        return "flat"
    change = (second - first) / abs(first)
    if change < -rel_tol:
        return "decreasing"
    if change > rel_tol:
        return "increasing"
    return "flat"


def summarize(log):
    total = len(log)
    maps = [r["metrics/mAP50-95(B)"] for r in log]

    best_i = max(range(total), key=lambda i: maps[i])
    best_map = maps[best_i]

    sm = smooth(maps)
    sm_best = max(sm)
    plateau_i = next(i for i in range(total) if sm[i] >= sm_best - NOISE)
    near_i = next(i for i in range(total) if sm[i] >= sm_best - NEAR)

    tail = max(1, total // 3)
    val_box_tail = [r["val/box_loss"] for r in log[-tail:]]
    trn_box_tail = [r["train/box_loss"] for r in log[-tail:]]
    map_tail = maps[-tail:]

    gap_early = mean([r["val/box_loss"] - r["train/box_loss"] for r in log[:tail]])
    gap_late = mean([r["val/box_loss"] - r["train/box_loss"] for r in log[-tail:]])

    return {
        "total_epochs":     total,
        "best_epoch":       int(log[best_i]["epoch"]),
        "best_map":         round(best_map, 4),
        "final_map":        round(maps[-1], 4),
        "plateau_epoch":    int(log[plateau_i]["epoch"]),
        "wasted_ratio":     round(1 - (plateau_i + 1) / total, 2),
        "near_best_epoch":  int(log[near_i]["epoch"]),
        "near_best_ratio":  round(1 - (near_i + 1) / total, 2),
        "trend_train_loss": trend(trn_box_tail),
        "trend_val_loss":   trend(val_box_tail),
        "trend_map":        trend(map_tail),
        "gap_early":        round(gap_early, 4),
        "gap_late":         round(gap_late, 4),
        "final_precision":  round(log[-1]["metrics/precision(B)"], 4),
        "final_recall":     round(log[-1]["metrics/recall(B)"], 4),
        "final_lr":         round(log[-1]["lr/pg0"], 8),
    }


def diagnose(s):
    msgs = []

    if s["near_best_ratio"] >= 0.4:
        msgs.append(
            f"Model reached within {NEAR} of its best score by epoch "
            f"{s['near_best_epoch']}; the remaining "
            f"{int(s['near_best_ratio'] * 100)}% of the budget yielded only "
            f"marginal gains."
        )

    if s["trend_val_loss"] == "increasing":
        msgs.append("Validation loss rises in the late stage: overfitting.")
    elif s["trend_train_loss"] == "decreasing" and s["trend_map"] == "flat":
        msgs.append(
            "Loss keeps decreasing while mAP stagnates in the late stage: "
            "optimization is not translating into detection performance."
        )

    if s["gap_late"] > s["gap_early"] * 1.5:
        if s["trend_val_loss"] != "increasing":
            msgs.append(
                "Train/val loss gap widens because train loss falls faster; "
                "validation loss is not rising, so this is not overfitting."
            )
        else:
            msgs.append(
                "Train/val loss gap widens and validation loss is rising: "
                "overfitting."
            )

    if s["final_precision"] - s["final_recall"] > 0.1:
        msgs.append(
            "Precision far exceeds recall: model is conservative, missing detections."
        )
    elif s["final_recall"] - s["final_precision"] > 0.1:
        msgs.append("Recall far exceeds precision: excessive false positives.")

    if not msgs:
        msgs.append("No obvious anomaly in the training process.")

    return msgs


def sample_curve(log, n=8):
    total = len(log)
    step = max(1, total // n)
    picked = list(range(0, total, step))
    if picked[-1] != total - 1:
        picked.append(total - 1)

    lines = ["epoch | train_box | val_box | mAP50-95 | lr"]
    for i in picked:
        r = log[i]
        lines.append(
            f"{int(r['epoch']):5d} | {r['train/box_loss']:9.3f} | "
            f"{r['val/box_loss']:7.3f} | {r['metrics/mAP50-95(B)']:8.4f} | "
            f"{r['lr/pg0']:.6f}"
        )
    return "\n".join(lines)


def to_text(log):
    s = summarize(log)
    parts = [
        f"Total epochs: {s['total_epochs']}",
        f"Best mAP50-95: {s['best_map']} (epoch {s['best_epoch']})",
        f"Final mAP50-95: {s['final_map']}",
        f"Reached within {NEAR} of best at epoch: {s['near_best_epoch']} "
        f"({int(s['near_best_ratio'] * 100)}% of budget spent on the last {NEAR})",
        f"Reached within {NOISE} of best at epoch: {s['plateau_epoch']}",
        f"Final precision / recall: {s['final_precision']} / {s['final_recall']}",
        f"Train-val loss gap: early {s['gap_early']} -> late {s['gap_late']}",
        "",
        "[Training curve samples]",
        sample_curve(log),
        "",
        "[Automated diagnosis]",
    ]
    parts += [f"- {m}" for m in diagnose(s)]
    return "\n".join(parts)


if __name__ == "__main__":
    log = read_log("C:/YOLO_agent_project/runs/baseline_s0/results.csv")
    print(to_text(log))