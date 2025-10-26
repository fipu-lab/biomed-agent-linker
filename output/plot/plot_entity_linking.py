import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = {
    "Approach": [
        "QuickUMLS",
        "SciSpacy",
        "SapBERT",
        "BioBERT",
        "Few-shot (Local)",
        "Agent (Local)",
        "Few-shot (Full)",
        "Agent (Full)",
    ],
    "N": [1000, 500, 1000, 1000, 1000, 500, 1000, 500],
    "Top-1 Accuracy": [0.205, 0.39, 0.518, 0.332, 0.351, 0.558, 0.441, 0.683],
    "Top-3 Accuracy": [0.252, 0.483, 0.619, 0.382, 0.372, 0.644, 0.498, 0.712],
    "Top-5 Accuracy": [0.254, 0.51, 0.647, 0.394, 0.389, 0.692, 0.506, 0.798],
}

df = pd.DataFrame(data)

colors = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]
hatches = [
    "//",
    "xx",
    "\\\\",
    "oo",
    "..",
    "++",
    "||",
    "--",
]


def plot_top_k_accuracies(df, title, filename):
    fig, ax = plt.subplots(figsize=(14, 7))

    approaches = df["Approach"]
    x = range(len(approaches))
    width = 0.25

    for i, (approach, top1, top3, top5) in enumerate(
        zip(
            df["Approach"],
            df["Top-1 Accuracy"],
            df["Top-3 Accuracy"],
            df["Top-5 Accuracy"],
        )
    ):
        bar1 = ax.bar(
            i - width,
            top1,
            width,
            color=colors[i % len(colors)],
            edgecolor="black",
            alpha=0.8,
            hatch=hatches[i % len(hatches)],
            zorder=3,
        )
        bar3 = ax.bar(
            i,
            top3,
            width,
            color=colors[i % len(colors)],
            edgecolor="black",
            alpha=0.6,
            hatch=hatches[i % len(hatches)],
            zorder=3,
        )
        bar5 = ax.bar(
            i + width,
            top5,
            width,
            color=colors[i % len(colors)],
            edgecolor="black",
            alpha=0.4,
            hatch=hatches[i % len(hatches)],
            zorder=3,
        )

    ax.set_ylabel("Accuracy", fontsize=18)
    ax.set_title(title, fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, rotation=45, ha="right", fontsize=13)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="gray", alpha=0.8, label="Accuracy@1"),
        Patch(facecolor="gray", alpha=0.6, label="Accuracy@3"),
        Patch(facecolor="gray", alpha=0.4, label="Accuracy@5"),
    ]
    ax.legend(handles=legend_elements, fontsize=14, loc="upper left")

    plt.tight_layout()
    plt.savefig(f"figures/{filename}.pdf", bbox_inches="tight")
    print(f"Saved figure to figures/{filename}.pdf")
    plt.show()


def plot_top1_accuracy_only(df, title, filename):
    plt.figure(figsize=(14, 7))

    approaches = df["Approach"]
    accuracies = df["Top-1 Accuracy"]

    for i, (approach, acc) in enumerate(zip(approaches, accuracies)):
        bar = plt.bar(
            approach,
            acc,
            color=colors[i % len(colors)],
            edgecolor="black",
            alpha=0.8,
            hatch=hatches[i % len(hatches)],
            zorder=3,
        )

        plt.text(
            i,
            acc + 0.015,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=16,
            rotation=0,
        )

    plt.ylabel("Accuracy@1", fontsize=18)
    plt.title(title, fontsize=24)
    plt.xticks(rotation=45, ha="right", fontsize=13)
    plt.yticks(fontsize=16)

    plt.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    plt.ylim(0, max(accuracies) * 1.15)

    plt.tight_layout()
    plt.savefig(f"figures/{filename}.pdf", bbox_inches="tight")
    print(f"Saved figure to figures/{filename}.pdf")
    plt.show()


plot_top_k_accuracies(
    df,
    "MedMentions ST21pv BEL: Accuracy@1, Accuracy@3, and Accuracy@5 Across Methods",
    "entity_linking_topk_accuracies",
)

plot_top1_accuracy_only(
    df,
    "MedMentions ST21pv BEL: Accuracy@1 Across Methods",
    "entity_linking_top1_accuracy",
)

print("\nSummary Statistics:")
print(df.to_string(index=False))
