import pandas as pd

data = {
    "Approach": [
        "QuickUMLS",
        "SciSpacy",
        "SapBERT",
        "BioBERT",
        "GPT-4o (No Tools, No Abstracts)",
        "GPT-4o agent (with Tools, No Abstracts)",
        "GPT-4o (No Tools, with Abstracts)",
        "GPT-4o agent (with Tools, with Abstracts)",
    ],
    "avg_time_ms": [4.63, 4656.55, 33.8, 32.43, 4066.88, 8603.97, 4098.12, 8890.76],
    "avg_tokens_in": [None, None, None, None, 887.69, 4784.46, 1093.43, 4998.34],
    "avg_tokens_out": [None, None, None, None, 186.03, 569.59, 183.40, 587.23],
}

df = pd.DataFrame(data)

GPT4O_INPUT_PRICE = 2.50
GPT4O_OUTPUT_PRICE = 10.00

df["qps"] = 1000 / df["avg_time_ms"]

df["price_1M_input"] = df["avg_tokens_in"].apply(
    lambda x: GPT4O_INPUT_PRICE if pd.notna(x) else None
)
df["price_1M_output"] = df["avg_tokens_out"].apply(
    lambda x: GPT4O_OUTPUT_PRICE if pd.notna(x) else None
)

df["avg_price_per_query"] = df.apply(
    lambda row: (
        (row["avg_tokens_in"] * row["price_1M_input"] / 1_000_000)
        + (row["avg_tokens_out"] * row["price_1M_output"] / 1_000_000)
        if pd.notna(row["avg_tokens_in"]) and pd.notna(row["avg_tokens_out"])
        else None
    ),
    axis=1,
)


def generate_latex_table(df):
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Performance Comparison of Entity Linking Approaches}")
    latex.append("\\label{tab:performance}")
    latex.append("\\begin{tabular}{lrr}")
    latex.append("\\hline")
    latex.append(
        "\\textbf{Approach} & \\textbf{Avg. Query Time (ms)} & \\textbf{QPS} \\\\"
    )
    latex.append("\\hline")

    for _, row in df.iterrows():
        approach = row["Approach"].replace("_", "\\_")
        avg_time = f"{row['avg_time_ms']:.2f}"
        qps = f"{row['qps']:.2f}"

        latex.append(f"{approach} & {avg_time} & {qps} \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


latex_table = generate_latex_table(df)
print(latex_table)
print("\n" + "=" * 80 + "\n")

output_file = "output/performance_cost_table.tex"
with open(output_file, "w") as f:
    f.write(latex_table)
print(f"LaTeX table saved to: {output_file}")

print("\nSummary Statistics:")
print(df.to_string(index=False))
