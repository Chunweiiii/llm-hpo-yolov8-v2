import matplotlib.pyplot as plt

COLUMNS = ["Method", "Best\nmAP50-95", "Iteration", "Duplicate\nrate", "Search space\ncoverage"]

ROWS = [
    ["Random Search", "0.7106", "1", "-", "-"],
    ["Optuna (TPE)",  "0.7111", "8", "-", "-"],
    ["Qwen2.5-Coder-7B", "0.7049", "7", "40% (4/10)", "3/9"],
    ["Claude Haiku 4.5", "0.7153", "5", "0%", "8/9"],
    ["Claude Sonnet 5", "0.7187", "5", "0%", "9/9"],
    ["Claude Opus 5", "0.7121", "4", "0%", "9/9"],
]

BASELINE_TEXT = "Default baseline (3-seed): mAP50-95 = 0.7024 \u00b1 0.0026"

HEADER_COLOR = "#3C3489"
HEADER_TEXT_COLOR = "#EEEDFE"
ROW_COLOR_EVEN = "#F1EFE8"
ROW_COLOR_ODD = "#FFFFFF"
HIGHLIGHT_COLOR = "#CEDDF6"
BEST_ROW_INDEX = 4

COL_WIDTHS = [0.26, 0.16, 0.14, 0.20, 0.24]


def main():
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axis("off")

    table = ax.table(
        cellText=ROWS,
        colLabels=COLUMNS,
        colWidths=COL_WIDTHS,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.3)

    for col in range(len(COLUMNS)):
        cell = table[0, col]
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(color=HEADER_TEXT_COLOR, weight="bold")

    for row in range(1, len(ROWS) + 1):
        row_color = ROW_COLOR_EVEN if row % 2 == 0 else ROW_COLOR_ODD
        for col in range(len(COLUMNS)):
            cell = table[row, col]
            if row - 1 == BEST_ROW_INDEX:
                cell.set_facecolor(HIGHLIGHT_COLOR)
            else:
                cell.set_facecolor(row_color)

    plt.title("Summary of hyperparameter search methods (10 iterations)", fontsize=13, pad=14)
    plt.figtext(0.5, 0.02, BASELINE_TEXT, ha="center", fontsize=9, color="#5F5E5A")

    plt.savefig("summary_table.png", dpi=300, bbox_inches="tight")
    print("已存檔: summary_table.png")


if __name__ == "__main__":
    main()