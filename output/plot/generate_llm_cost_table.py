import pandas as pd

data = {
    "Approach": [
        "Few-shot (Local)",
        "Agent (Local)",
        "Few-shot (Full)",
        "Agent (Full)",
    ],
    "avg_tokens_in": [887.69, 4784.46, 1093.43, 4998.34],
    "avg_tokens_out": [186.03, 569.59, 183.40, 587.23],
}

df = pd.DataFrame(data)

GPT4O_INPUT_PRICE = 2.50
GPT4O_OUTPUT_PRICE = 10.00

df["avg_price_per_query"] = (df["avg_tokens_in"] * GPT4O_INPUT_PRICE / 1_000_000) + (
    df["avg_tokens_out"] * GPT4O_OUTPUT_PRICE / 1_000_000
)


def generate_latex_table(df, input_price, output_price):
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(
        f"\\caption{{Token Usage and Cost Analysis of LLM-based Approaches. "
        f"Pricing: \\${input_price:.2f} per 1M input tokens, \\${output_price:.2f} per 1M output tokens.}}"
    )
    latex.append("\\label{tab:llm_cost}")
    latex.append("\\begin{tabular}{lrrr}")
    latex.append("\\hline")
    latex.append(
        "\\textbf{Approach} & \\textbf{Avg. Tokens In} & \\textbf{Avg. Tokens Out} & \\textbf{Avg. Price per Query (\$)} \\\\"
    )
    latex.append("\\hline")

    for _, row in df.iterrows():
        approach = row["Approach"].replace("_", "\\_")
        tokens_in = f"{row['avg_tokens_in']:.2f}"
        tokens_out = f"{row['avg_tokens_out']:.2f}"
        price = f"{row['avg_price_per_query']:.6f}"

        latex.append(f"{approach} & {tokens_in} & {tokens_out} & {price} \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


latex_table = generate_latex_table(df, GPT4O_INPUT_PRICE, GPT4O_OUTPUT_PRICE)
print(latex_table)
print("\n" + "=" * 80 + "\n")

output_file = "output/llm_cost_table.tex"
with open(output_file, "w") as f:
    f.write(latex_table)
print(f"LaTeX table saved to: {output_file}")

print("\nSummary Statistics:")
print(df.to_string(index=False))
print(
    f"\nPricing: ${GPT4O_INPUT_PRICE} per 1M input tokens, ${GPT4O_OUTPUT_PRICE} per 1M output tokens"
)
