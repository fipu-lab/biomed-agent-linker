import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("output/summary_main.xlsx")

df = pd.DataFrame(data)

pareto_front_correct = df[df["model"].isin(["gpt-4o", "llama3.3-70b", "deepseek-v3"])]

pareto_front_correct = pareto_front_correct.sort_values(by="price_total")


plt.figure(figsize=(10, 6))
plt.xlim(0, 0.008)

plt.scatter(
    df["price_total"],
    df["acc_mean"],
    color="blue",
    edgecolor="black",
    alpha=0.7,
    s=100,
    label="Models",
    zorder=3,
)

for i, model in enumerate(df["model"]):
    plt.text(
        df["price_total"][i] + 10e-5,
        df["acc_mean"][i] - 0.005,
        model,
        fontsize=10,
        ha="left",
        va="bottom",
        zorder=4,
    )

plt.scatter(
    pareto_front_correct["price_total"],
    pareto_front_correct["acc_mean"],
    color="red",
    edgecolor="black",
    s=120,
    label="Pareto Front",
    zorder=4,
)

plt.plot(
    pareto_front_correct["price_total"],
    pareto_front_correct["acc_mean"],
    color="red",
    linestyle="--",
    linewidth=1.5,
    zorder=2,
)

plt.xlabel("Total Cost ($)", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.title("Cost vs. Accuracy with Pareto Front", fontsize=16)
plt.grid(axis="both", linestyle=":", alpha=0.5, zorder=0)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig("figures/scatter_pareto_front_final.pdf")
plt.show()
