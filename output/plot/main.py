import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

# Load both datasets
try:
    df_main = pd.read_excel("output/old/summary_main_v2.xlsx")
    print(f"Loaded main dataset with {len(df_main)} models")
except FileNotFoundError:
    print("Main dataset summary not found")
    df_main = None

df_nl2sql = pd.read_excel("output/summary_questions_v2_100.xlsx")
print(f"Loaded NL2SQL dataset with {len(df_nl2sql)} models")
hatches = [
    "//",
    "xx",
    "\\",
    "oo",
    "..",
    "++",
    "||",
    "--",
    "**",
]
colors = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
    "#1f78b4",
]  # Color-blind friendly palette extended for 9 models

# Define the desired model order for NL2SQL
MODEL_ORDER = [
    "deepseek-v3",
    "deepseek-v3.1",
    "llama3.3-70b",
    "qwen2p5-vl-32b-instruct",
    "mixtral-8x22b-instruct",
    "bio-mistral-7b",
    "gpt-4o",
    "gpt-5",
    "gpt-oss-20b",
]

# Display names for better readability
MODEL_DISPLAY_NAMES = {
    "deepseek-v3": "DeepSeek V3",
    "deepseek-v3.1": "DeepSeek V3.1",
    "llama3.3-70b": "Llama 3.3-70B",
    "qwen2p5-vl-32b-instruct": "Qwen2.5-32B-instruct",
    "mixtral-8x22b-instruct": "Mixtral-8x22B-instruct",
    "bio-mistral-7b": "BioMistral-7B",
    "gpt-4o": "GPT-4o",
    "gpt-5": "GPT-5",
    "gpt-oss-20b": "GPT-OSS-20B",
}


def plot_accuracy(df, title, filename, exclude_models=None, sort_order=None):
    """Plot accuracy for a given dataset"""
    if df is None:
        print(f"Skipping {title} - no data available")
        return

    plt.figure(figsize=(12, 6))
    models_to_plot = []
    accuracies = []

    # Filter models
    filtered_df = df.copy()
    if exclude_models:
        filtered_df = filtered_df[~filtered_df["model"].isin(exclude_models)]

    # Sort models according to specified order
    if sort_order:
        filtered_df["sort_idx"] = filtered_df["model"].map(
            {model: idx for idx, model in enumerate(sort_order)}
        )
        filtered_df = filtered_df.sort_values("sort_idx")
        filtered_df = filtered_df.drop("sort_idx", axis=1)

    for i, (model, acc) in enumerate(
        zip(filtered_df["model"], filtered_df["acc_mean"])
    ):
        models_to_plot.append(model)
        accuracies.append(acc)

    bars = []
    for i, (model, acc) in enumerate(zip(models_to_plot, accuracies)):
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        bar = plt.bar(
            display_name,
            acc,
            color=colors[i % len(colors)],
            edgecolor="black",
            alpha=0.8,
            hatch=hatches[i % len(hatches)],
            zorder=3,
        )
        bars.append(bar[0])

        # Add value labels on top of bars
        plt.text(
            i,
            acc + 0.01,  # Position slightly above the bar
            f"{acc:.3f}",  # Format to 3 decimal places
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            rotation=0,
        )

    plt.ylabel("Execution Accuracy (EX)", fontsize=16)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=14)
    plt.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    # Adjust y-axis to make room for text labels
    if accuracies:
        plt.ylim(0, max(accuracies) * 1.15)  # Add 15% padding for labels

    plt.tight_layout()
    plt.savefig(f"figures/{filename}.pdf", bbox_inches="tight")
    plt.show()


def plot_cost(df, title, filename, exclude_models=None, sort_order=None):
    """Plot total cost for a given dataset"""
    if df is None:
        print(f"Skipping {title} - no data available")
        return

    plt.figure(figsize=(12, 6))
    models_to_plot = []
    costs = []

    # Filter models
    filtered_df = df.copy()
    if exclude_models:
        filtered_df = filtered_df[~filtered_df["model"].isin(exclude_models)]

    # Sort models according to specified order
    if sort_order:
        filtered_df["sort_idx"] = filtered_df["model"].map(
            {model: idx for idx, model in enumerate(sort_order)}
        )
        filtered_df = filtered_df.sort_values("sort_idx")
        filtered_df = filtered_df.drop("sort_idx", axis=1)

    for i, (model, cost) in enumerate(
        zip(filtered_df["model"], filtered_df["price_total"])
    ):
        models_to_plot.append(model)
        costs.append(cost)

    for i, (model, cost) in enumerate(zip(models_to_plot, costs)):
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        plt.bar(
            display_name,
            cost,
            color=colors[i % len(colors)],
            edgecolor="black",
            alpha=0.8,
            hatch=hatches[i % len(hatches)],
            zorder=3,
        )

        # Add value labels on top of bars
        plt.text(
            i,
            cost
            + max(costs) * 0.02,  # Position slightly above the bar (2% of max cost)
            f"${cost:.4f}",  # Format as currency with 4 decimal places
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            rotation=0,
        )

    plt.ylabel("Total Cost ($)", fontsize=16)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=14)
    plt.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    # Adjust y-axis to make room for text labels
    if costs:
        plt.ylim(0, max(costs) * 1.20)  # Add 20% padding for labels

    plt.tight_layout()
    plt.savefig(f"figures/{filename}.pdf", bbox_inches="tight")
    plt.show()


def plot_avg_cost_per_1000_tokens(
    df, title, filename, exclude_models=None, sort_order=None
):
    """Plot average cost per 1000 tokens for a given dataset"""
    if df is None:
        print(f"Skipping {title} - no data available")
        return

    plt.figure(figsize=(12, 6))
    models_to_plot = []
    avg_costs_1000_tokens = []

    # Filter models
    filtered_df = df.copy()
    if exclude_models:
        filtered_df = filtered_df[~filtered_df["model"].isin(exclude_models)]

    # Sort models according to specified order
    if sort_order:
        filtered_df["sort_idx"] = filtered_df["model"].map(
            {model: idx for idx, model in enumerate(sort_order)}
        )
        filtered_df = filtered_df.sort_values("sort_idx")
        filtered_df = filtered_df.drop("sort_idx", axis=1)

    # The price_total column already contains the correct cost per query
    # Cost per 1000 queries = price_total * 1000
    filtered_df["cost_per_1000_queries"] = filtered_df["price_total"] * 1000

    for i, (model, cost_per_1000_queries) in enumerate(
        zip(filtered_df["model"], filtered_df["cost_per_1000_queries"])
    ):
        models_to_plot.append(model)
        avg_costs_1000_tokens.append(cost_per_1000_queries)

    for i, (model, cost_per_1000_queries) in enumerate(
        zip(models_to_plot, avg_costs_1000_tokens)
    ):
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        plt.bar(
            display_name,
            cost_per_1000_queries,
            color=colors[i % len(colors)],
            edgecolor="black",
            alpha=0.8,
            hatch=hatches[i % len(hatches)],
            zorder=3,
        )

        # Add value labels on top of bars (per 1000 queries only)
        plt.text(
            i,
            cost_per_1000_queries
            + max(avg_costs_1000_tokens) * 0.02,  # Position slightly above the bar
            f"${cost_per_1000_queries:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            rotation=0,
        )

    plt.ylabel("Average Cost per 1000 Queries ($)", fontsize=16)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=14)
    plt.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    # Force decimal notation for y-axis (prevent scientific notation)
    plt.ticklabel_format(axis="y", style="plain", useOffset=False)

    # Adjust y-axis to make room for text labels
    if avg_costs_1000_tokens:
        plt.ylim(0, max(avg_costs_1000_tokens) * 1.20)  # Add 20% padding for labels

    plt.tight_layout()
    plt.savefig(f"figures/{filename}.pdf", bbox_inches="tight")
    plt.show()


def plot_pareto_front_cost_accuracy(df, title, filename, exclude_models=None):
    """Plot Pareto front for cost vs accuracy"""
    if df is None:
        print(f"Skipping {title} - no data available")
        return

    plt.figure(figsize=(12, 8))

    # Filter models
    filtered_df = df.copy()
    if exclude_models:
        filtered_df = filtered_df[~filtered_df["model"].isin(exclude_models)]

    # Use cost per query (price_total already contains this)
    filtered_df["cost_per_query"] = filtered_df["price_total"]

    # Sort by cost for Pareto front calculation
    filtered_df_sorted = filtered_df.sort_values(by="cost_per_query")

    # Calculate Pareto front
    pareto_front = []
    max_accuracy = 0

    for _, row in filtered_df_sorted.iterrows():
        if row["acc_mean"] > max_accuracy:
            pareto_front.append(row)
            max_accuracy = row["acc_mean"]

    pareto_front_df = pd.DataFrame(pareto_front)

    # Plot all models as scatter points
    plt.scatter(
        filtered_df["cost_per_query"],
        filtered_df["acc_mean"],
        color="lightblue",
        edgecolor="black",
        alpha=0.7,
        s=100,
        label="All Models",
        zorder=3,
    )

    # Add model labels
    for i, row in filtered_df.iterrows():
        plt.text(
            row["cost_per_query"] + 0.000002,  # Minimal offset to the right
            row["acc_mean"] - 0.01,  # Below the points
            MODEL_DISPLAY_NAMES.get(row["model"], row["model"]),
            fontsize=8,
            ha="left",  # Left-align so text starts from the point
            va="top",  # Align to top of text (since we're below)
            zorder=4,
        )

    # Plot Pareto front
    if len(pareto_front_df) > 1:
        plt.scatter(
            pareto_front_df["cost_per_query"],
            pareto_front_df["acc_mean"],
            color="red",
            edgecolor="black",
            s=120,
            label="Pareto Front",
            zorder=4,
        )

        plt.plot(
            pareto_front_df["cost_per_query"],
            pareto_front_df["acc_mean"],
            color="red",
            linestyle="--",
            linewidth=2,
            zorder=2,
        )

    plt.xlabel("Average Cost per Query ($)", fontsize=14)
    plt.ylabel("Execution Accuracy (EX)", fontsize=14)
    plt.title("NL2SQL: " + title, fontsize=16)
    plt.grid(axis="both", linestyle=":", alpha=0.5, zorder=0)
    plt.legend(fontsize=12)

    # Set reasonable axis limits
    if len(filtered_df) > 0:
        x_max = filtered_df["cost_per_query"].max() * 1.1
        y_min = max(0, filtered_df["acc_mean"].min() - 0.05)
        y_max = min(1, filtered_df["acc_mean"].max() + 0.05)
        plt.xlim(0, x_max)
        plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f"figures/{filename}.pdf", bbox_inches="tight")
    plt.show()


# Plot NL2SQL Execution Accuracy (9 models) - sorted
plot_accuracy(
    df_nl2sql,
    "NL2SQL Execution Accuracy per LLM (3-Run Average)",
    "accuracy_nl2sql",
    sort_order=MODEL_ORDER,
)

# Plot NL2SQL Total Cost (9 models) - sorted
plot_cost(
    df_nl2sql, "NL2SQL Total Cost (9 Models)", "cost_nl2sql", sort_order=MODEL_ORDER
)

# Plot NL2SQL Average Cost per 1000 Queries (8 models, excluding BioMistral) - sorted
plot_avg_cost_per_1000_tokens(
    df_nl2sql,
    "NL2SQL: Average Cost per 1000 Generated Queries (8 Models, 3-Run Average)",
    "avg_cost_per_1000_queries_nl2sql",
    exclude_models=["bio-mistral-7b"],
    sort_order=MODEL_ORDER,
)

# Plot Cost vs Accuracy Pareto Front (8 models, excluding BioMistral)
plot_pareto_front_cost_accuracy(
    df_nl2sql,
    "Cost vs Accuracy Pareto Front (8 Models, Excluding BioMistral)",
    "pareto_front_cost_accuracy_nl2sql",
    exclude_models=["bio-mistral-7b"],
)

# Plot Main Dataset (if available)
if df_main is not None:
    plot_accuracy(
        df_main,
        "Main Dataset Execution Accuracy",
        "accuracy_main",
        exclude_models=["deepseek-r1"],
    )
    plot_cost(
        df_main, "Main Dataset Total Cost", "cost_main", exclude_models=["deepseek-r1"]
    )
