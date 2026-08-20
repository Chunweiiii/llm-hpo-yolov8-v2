import matplotlib.pyplot as plt

TITLE = "Traffic Light Detection — Filtered & Augmented Dataset (937 images)"
THRESHOLD = 0.008  # 2\u03c3 noise threshold, derived from new dataset's 3-seed baseline std (0.0040)

# (Method, display_value, delta_value_or_None, delta_display, note)
GROUPS = [
    ("Non-LLM baselines", [
        ("Default",       "0.6463 \u00b1 0.0040", None,   "\u2013",              "\u2013"),
        ("Random Search", "0.6730",               0.0289, "+0.0289 (+4.5%)",     "iter 7"),
        ("Optuna",        "0.6707",               0.0266, "+0.0266 (+4.1%)",     "iter 1"),
    ]),
    ("LLM-based methods", [
        ("Qwen2.5-7B", "0.6564", 0.0123, "+0.0123 (+1.9%)", "iter 2, dup 30%"),
        ("Haiku 4.5",  "0.6707", 0.0266, "+0.0266 (+4.1%)", "iter 1, dup 0%"),
        ("Sonnet 5",   "0.6642", 0.0201, "+0.0201 (+3.1%)", "iter 1, dup 0%"),
        ("Opus 5",     "0.6766", 0.0325, "+0.0325 (+5.0%)", "iter 7, dup 0%"),
    ]),
]

COLUMNS = ["Method", "mAP50-95", "\u0394mAP vs Default", "Notes"]


def is_significant(delta):
    if delta is None:
        return False
    return delta >= THRESHOLD


def main():
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.axis("off")

    y = 0.97
    line_h = 0.068

    ax.text(0.5, y, TITLE, ha="center", fontsize=11.5, fontweight="bold")
    y -= line_h * 1.2
    ax.axhline(y + line_h * 0.5, color="black", linewidth=1.5, xmin=0.02, xmax=0.98)

    col_x = [0.06, 0.30, 0.56, 0.85]
    aligns = ["left", "center", "center", "center"]
    for cx, label, al in zip(col_x, COLUMNS, aligns):
        ax.text(cx, y, label, fontsize=11, fontweight="bold", ha=al)
    y -= line_h * 0.85
    ax.axhline(y + line_h * 0.5, color="black", linewidth=1, xmin=0.02, xmax=0.98)

    for group_name, rows in GROUPS:
        y -= line_h * 0.25
        for name, mapval, delta, delta_disp, note in rows:
            sig = is_significant(delta)
            weight = "bold" if sig else "normal"
            ax.text(col_x[0], y, name, fontsize=10.5, ha="left", fontweight=weight)
            ax.text(col_x[1], y, mapval, fontsize=10.5, ha="center", fontweight=weight)
            ax.text(col_x[2], y, delta_disp, fontsize=10.5, ha="center", fontweight=weight)
            ax.text(col_x[3], y, note, fontsize=9, ha="center", color="#555555")
            y -= line_h
        y -= line_h * 0.1
        ax.axhline(y + line_h * 0.5, color="#CCCCCC", linewidth=0.8, xmin=0.02, xmax=0.98)

    ax.axhline(y + line_h * 0.5, color="black", linewidth=1.5, xmin=0.02, xmax=0.98)

    y -= line_h * 0.55
    ax.text(
        0.5, y,
        f"Bold rows indicate improvement exceeding the noise threshold (\u00b1{THRESHOLD:.3f}, ~2\u03c3).",
        fontsize=8.5, ha="center", color="#777777", style="italic",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(y - 0.05, 1.02)

    plt.savefig("results_table_v2.png", dpi=300, bbox_inches="tight")
    print("\u5df2\u5b58\u6a94: results_table_v2.png")


if __name__ == "__main__":
    main()