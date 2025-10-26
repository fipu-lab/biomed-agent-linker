import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = {
    "Approach": [
        "Few-shot (Local)",
        "Few-shot (Full)",
        "Agent (Local)",
        "Agent (Full)",
    ],
    "Full Name": [
        "Few-shot prompting\n(Local context ±100 tokens)",
        "Few-shot prompting\n(Full abstract)",
        "Agent with tool calling\n(Local context ±100 tokens)",
        "Agent with tool calling\n(Full abstract)",
    ],
    "Top-1 Accuracy": [0.351, 0.441, 0.558, 0.683],
    "avg_price_per_query": [0.004080, 0.004568, 0.017657, 0.018368],
}

df = pd.DataFrame(data)

df_sorted = df.sort_values(by="avg_price_per_query")

pareto_front = []
max_accuracy = 0

for _, row in df_sorted.iterrows():
    if row["Top-1 Accuracy"] > max_accuracy:
        pareto_front.append(row)
        max_accuracy = row["Top-1 Accuracy"]

pareto_front_df = pd.DataFrame(pareto_front)

plt.figure(figsize=(12, 8))

plt.scatter(
    df["avg_price_per_query"],
    df["Top-1 Accuracy"],
    color="lightblue",
    edgecolor="black",
    alpha=0.7,
    s=150,
    label="All Approaches",
    zorder=3,
)

for i, row in df.iterrows():
    plt.text(
        row["avg_price_per_query"] + 0.0003,
        row["Top-1 Accuracy"] - 0.015,
        row["Approach"],
        fontsize=16,
        ha="left",
        va="top",
        zorder=4,
    )

plt.scatter(
    pareto_front_df["avg_price_per_query"],
    pareto_front_df["Top-1 Accuracy"],
    color="red",
    edgecolor="black",
    s=180,
    label="Pareto Front",
    zorder=4,
)

plt.plot(
    pareto_front_df["avg_price_per_query"],
    pareto_front_df["Top-1 Accuracy"],
    color="red",
    linestyle="--",
    linewidth=2,
    zorder=2,
)

plt.xlabel("Average Cost per Query ($)", fontsize=18)
plt.ylabel("Accuracy@1", fontsize=18)
plt.title(
    "MedMentions ST21pv BEL: Cost vs. Accuracy Trade-off with Pareto Front",
    fontsize=24,
)
plt.grid(axis="both", linestyle=":", alpha=0.5, zorder=0)
plt.legend(fontsize=15, loc="lower right")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

x_min = 0
x_max = df["avg_price_per_query"].max() * 1.15
y_min = max(0, df["Top-1 Accuracy"].min() - 0.05)
y_max = min(1, df["Top-1 Accuracy"].max() + 0.05)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.savefig("figures/llm_pareto_front.pdf", bbox_inches="tight")
print("Saved figure to figures/llm_pareto_front.pdf")
plt.show()

print("\nPareto Front Analysis:")
print("=" * 80)
print(f"Approaches on Pareto Front: {len(pareto_front_df)} out of {len(df)}")
print("\nPareto-optimal approaches (sorted by cost):")
for _, row in pareto_front_df.iterrows():
    print(
        f"  - {row['Approach']:20s}: Acc@1={row['Top-1 Accuracy']:.3f}, Cost=${row['avg_price_per_query']:.6f}"
    )

print("\n" + "=" * 80)
print("Trade-off Analysis:")
print("=" * 80)
print(
    f"\nMost cost-effective: {df.loc[df['avg_price_per_query'].idxmin(), 'Approach']}"
)
print(
    f"  Cost: ${df['avg_price_per_query'].min():.6f}, Accuracy: {df.loc[df['avg_price_per_query'].idxmin(), 'Top-1 Accuracy']:.3f}"
)
print(f"\nHighest accuracy: {df.loc[df['Top-1 Accuracy'].idxmax(), 'Approach']}")
print(
    f"  Cost: ${df.loc[df['Top-1 Accuracy'].idxmax(), 'avg_price_per_query']:.6f}, Accuracy: {df['Top-1 Accuracy'].max():.3f}"
)

cost_diff = (
    df.loc[df["Top-1 Accuracy"].idxmax(), "avg_price_per_query"]
    - df["avg_price_per_query"].min()
)
acc_gain = (
    df["Top-1 Accuracy"].max()
    - df.loc[df["avg_price_per_query"].idxmin(), "Top-1 Accuracy"]
)
print(
    f"\nCost increase for max accuracy: ${cost_diff:.6f} ({cost_diff/df['avg_price_per_query'].min()*100:.1f}% increase)"
)
print(
    f"Accuracy gain: +{acc_gain:.3f} ({acc_gain/df.loc[df['avg_price_per_query'].idxmin(), 'Top-1 Accuracy']*100:.1f}% improvement)"
)
