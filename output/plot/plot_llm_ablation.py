import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data = {
    "Approach": [
        "Few-shot prompting",
        "Few-shot prompting",
        "Agent with tool calling",
        "Agent with tool calling",
    ],
    "Context": [
        "Local context (±100 tokens)",
        "Full abstract",
        "Local context (±100 tokens)",
        "Full abstract",
    ],
    "Top-1 Accuracy": [0.351, 0.441, 0.558, 0.683],
    "Top-3 Accuracy": [0.372, 0.498, 0.644, 0.712],
    "Top-5 Accuracy": [0.389, 0.506, 0.692, 0.798],
}

df = pd.DataFrame(data)

colors = ["#1b9e77", "#d95f02"]
hatches = ["//", "xx"]


def plot_ablation_grouped_bars(df, title, filename):
    fig, ax = plt.subplots(figsize=(12, 7))

    contexts = ["Local context (±100 tokens)", "Full abstract"]
    approaches = ["Few-shot prompting", "Agent with tool calling"]

    x = np.arange(len(contexts))
    width = 0.35

    for i, approach in enumerate(approaches):
        approach_data = df[df["Approach"] == approach]
        accuracies = []
        for context in contexts:
            acc = approach_data[approach_data["Context"] == context][
                "Top-1 Accuracy"
            ].values[0]
            accuracies.append(acc)

        bars = ax.bar(
            x + i * width,
            accuracies,
            width,
            label=approach,
            color=colors[i],
            edgecolor="black",
            alpha=0.8,
            hatch=hatches[i],
            zorder=3,
        )

        for j, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontsize=17,
            )

    ax.set_ylabel("Accuracy@1", fontsize=18)
    ax.set_title(title, fontsize=24)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(contexts, fontsize=14)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(fontsize=15, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.set_ylim(0, 0.8)

    plt.tight_layout()
    plt.savefig(f"figures/{filename}.pdf", bbox_inches="tight")
    print(f"Saved figure to figures/{filename}.pdf")
    plt.show()


def plot_ablation_all_topk(df, title, filename):
    fig, ax = plt.subplots(figsize=(14, 7))

    contexts = ["Local context (±100 tokens)", "Full abstract"]
    approaches = ["Few-shot prompting", "Agent with tool calling"]
    metrics = ["Top-1 Accuracy", "Top-3 Accuracy", "Top-5 Accuracy"]

    x = np.arange(len(contexts))
    total_width = 0.8
    n_bars = len(approaches) * len(metrics)
    width = total_width / n_bars

    alphas = [0.9, 0.6, 0.3]

    bar_index = 0
    for i, approach in enumerate(approaches):
        for j, metric in enumerate(metrics):
            approach_data = df[df["Approach"] == approach]
            accuracies = []
            for context in contexts:
                acc = approach_data[approach_data["Context"] == context][metric].values[
                    0
                ]
                accuracies.append(acc)

            offset = bar_index * width - total_width / 2 + width / 2
            bars = ax.bar(
                x + offset,
                accuracies,
                width,
                label=f"{approach} - {metric.replace(' Accuracy', '')}",
                color=colors[i],
                edgecolor="black",
                alpha=alphas[j],
                hatch=hatches[i],
                zorder=3,
            )

            bar_index += 1

    ax.set_ylabel("Accuracy", fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(contexts, fontsize=13)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(fontsize=10, loc="upper left", ncol=2)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.set_ylim(0, 0.9)

    plt.tight_layout()
    plt.savefig(f"figures/{filename}.pdf", bbox_inches="tight")
    print(f"Saved figure to figures/{filename}.pdf")
    plt.show()


def plot_ablation_line_chart(df, title, filename):
    fig, ax = plt.subplots(figsize=(12, 7))

    contexts = ["Local context\n(±100 tokens)", "Full abstract"]
    approaches = ["Few-shot prompting", "Agent with tool calling"]

    for i, approach in enumerate(approaches):
        approach_data = df[df["Approach"] == approach].sort_values("Context")
        accuracies = approach_data["Top-1 Accuracy"].values

        ax.plot(
            contexts,
            accuracies,
            marker="o",
            markersize=12,
            linewidth=3,
            label=approach,
            color=colors[i],
            zorder=3,
        )

        for j, (context, acc) in enumerate(zip(contexts, accuracies)):
            ax.text(
                j,
                acc + 0.02,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color=colors[i],
            )

    ax.set_ylabel("Accuracy@1", fontsize=16)
    ax.set_xlabel("Context Type", fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(fontsize=13, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.set_ylim(0.3, 0.75)

    plt.tight_layout()
    plt.savefig(f"figures/{filename}.pdf", bbox_inches="tight")
    print(f"Saved figure to figures/{filename}.pdf")
    plt.show()


print("Ablation Study Data:")
print(df.to_string(index=False))
print("\n" + "=" * 80 + "\n")

plot_ablation_grouped_bars(
    df,
    "MedMentions ST21pv BEL LLM Ablation Study: Impact of Context Type and Tool Calling",
    "llm_ablation_grouped",
)

plot_ablation_line_chart(
    df,
    "LLM Ablation Study: Impact of Context Type and Tool Calling",
    "llm_ablation_line",
)

plot_ablation_all_topk(
    df,
    "LLM Ablation Study: All Top-k Metrics",
    "llm_ablation_all_topk",
)

print("\n" + "=" * 80)
print("Impact Analysis:")
print("=" * 80)

local_fewshot = df[
    (df["Approach"] == "Few-shot prompting")
    & (df["Context"] == "Local context (±100 tokens)")
]["Top-1 Accuracy"].values[0]
full_fewshot = df[
    (df["Approach"] == "Few-shot prompting") & (df["Context"] == "Full abstract")
]["Top-1 Accuracy"].values[0]
local_agent = df[
    (df["Approach"] == "Agent with tool calling")
    & (df["Context"] == "Local context (±100 tokens)")
]["Top-1 Accuracy"].values[0]
full_agent = df[
    (df["Approach"] == "Agent with tool calling") & (df["Context"] == "Full abstract")
]["Top-1 Accuracy"].values[0]

print(f"\nImpact of Full Abstract (vs. Local Context):")
print(
    f"  - Few-shot prompting: +{(full_fewshot - local_fewshot):.3f} ({((full_fewshot/local_fewshot - 1) * 100):.1f}% improvement)"
)
print(
    f"  - Agent with tool calling: +{(full_agent - local_agent):.3f} ({((full_agent/local_agent - 1) * 100):.1f}% improvement)"
)

print(f"\nImpact of Tool Calling (vs. Few-shot):")
print(
    f"  - Local context: +{(local_agent - local_fewshot):.3f} ({((local_agent/local_fewshot - 1) * 100):.1f}% improvement)"
)
print(
    f"  - Full abstract: +{(full_agent - full_fewshot):.3f} ({((full_agent/full_fewshot - 1) * 100):.1f}% improvement)"
)
