"""
Few-shot examples for the code generation LLM.

Provides concrete input→output examples that guide the model toward
the expected coding style, output format, and chart aesthetics.

Optimization for Qwen2.5:7B:
    - Fewer examples (6 instead of 10+) to save context tokens
    - Each example covers a distinct pattern
    - Chart examples include dark theme styling
    - Covers bar, histogram, scatter, correlation, box, violin patterns
"""

from __future__ import annotations

FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "question": "What is the average revenue by category?",
        "code": (
            "# Calculate average revenue grouped by category\n"
            "result = df.groupby('category')['revenue'].mean()"
            ".sort_values(ascending=False).round(2)"
        ),
    },
    {
        "question": "Show me a bar chart of total sales by month",
        "code": (
            "import matplotlib.pyplot as plt\n"
            "\n"
            "# Convert date and aggregate monthly sales\n"
            "df['date'] = pd.to_datetime(df['date'])\n"
            "monthly = df.groupby(df['date'].dt.to_period('M'))['revenue'].sum()\n"
            "monthly.index = monthly.index.astype(str)\n"
            "\n"
            "# Professional dark-themed bar chart\n"
            "palette = ['#8B5CF6','#06D6A0','#F72585','#FFD166','#4CC9F0','#A78BFA','#EF476F','#118AB2']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "fig.patch.set_facecolor('#0f0f1a')\n"
            "ax.set_facecolor('#151528')\n"
            "bars = ax.bar(range(len(monthly)), monthly.values,\n"
            "              color=[palette[i % len(palette)] for i in range(len(monthly))],\n"
            "              width=0.6, edgecolor='none', zorder=3)\n"
            "ax.bar_label(bars, fmt='$%.0f', fontsize=8, color='#E2E8F0', padding=4)\n"
            "ax.set_xticks(range(len(monthly)))\n"
            "ax.set_xticklabels(monthly.index, rotation=45, ha='right', fontsize=9, color='#E2E8F0')\n"
            "ax.set_title('Total Sales by Month', fontsize=15, fontweight='bold', color='#E2E8F0', pad=15)\n"
            "ax.set_ylabel('Revenue ($)', fontsize=11, color='#E2E8F0', labelpad=8)\n"
            "ax.tick_params(colors='#E2E8F0', labelsize=9)\n"
            "ax.spines['top'].set_visible(False)\n"
            "ax.spines['right'].set_visible(False)\n"
            "ax.spines['left'].set_color('#2D2D44')\n"
            "ax.spines['bottom'].set_color('#2D2D44')\n"
            "ax.yaxis.grid(True, alpha=0.15, color='#4A4A6A', linestyle='--', zorder=0)\n"
            "ax.xaxis.grid(False)\n"
            "plt.tight_layout(pad=1.5)\n"
            "plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')\n"
            "plt.close('all')\n"
            "\n"
            "result = monthly"
        ),
    },
    {
        "question": "Show a histogram of the price distribution",
        "code": (
            "import matplotlib.pyplot as plt\n"
            "\n"
            "# Professional histogram with statistical annotations\n"
            "values = pd.to_numeric(df['price'], errors='coerce').dropna()\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "fig.patch.set_facecolor('#0f0f1a')\n"
            "ax.set_facecolor('#151528')\n"
            "ax.hist(values, bins=30, color='#8B5CF6', edgecolor='#151528', linewidth=0.5, alpha=0.9, zorder=3)\n"
            "ax.axvline(values.mean(), color='#EF476F', linestyle='--', linewidth=1.5, label=f'Mean: ${values.mean():,.2f}', zorder=4)\n"
            "ax.axvline(values.median(), color='#06D6A0', linestyle='--', linewidth=1.5, label=f'Median: ${values.median():,.2f}', zorder=4)\n"
            "ax.set_title('Price Distribution', fontsize=15, fontweight='bold', color='#E2E8F0', pad=15)\n"
            "ax.set_xlabel('Price ($)', fontsize=11, color='#E2E8F0', labelpad=8)\n"
            "ax.set_ylabel('Frequency', fontsize=11, color='#E2E8F0', labelpad=8)\n"
            "ax.tick_params(colors='#E2E8F0', labelsize=9)\n"
            "ax.spines['top'].set_visible(False)\n"
            "ax.spines['right'].set_visible(False)\n"
            "ax.spines['left'].set_color('#2D2D44')\n"
            "ax.spines['bottom'].set_color('#2D2D44')\n"
            "ax.yaxis.grid(True, alpha=0.15, color='#4A4A6A', linestyle='--', zorder=0)\n"
            "ax.xaxis.grid(False)\n"
            "ax.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#2D2D44', labelcolor='#E2E8F0')\n"
            "plt.tight_layout(pad=1.5)\n"
            "plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')\n"
            "plt.close('all')\n"
            "\n"
            "result = f'Mean: ${values.mean():,.2f}, Median: ${values.median():,.2f}, Std: ${values.std():,.2f}'"
        ),
    },
    {
        "question": "Show the correlation matrix for all numeric columns",
        "code": (
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "\n"
            "# Correlation heatmap with masked upper triangle\n"
            "num_df = df.select_dtypes(include='number')\n"
            "corr = num_df.corr()\n"
            "mask = np.triu(np.ones_like(corr, dtype=bool), k=1)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8, 5))\n"
            "fig.patch.set_facecolor('#0f0f1a')\n"
            "ax.set_facecolor('#151528')\n"
            "sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='magma',\n"
            "            vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,\n"
            "            annot_kws={'fontsize': 9, 'color': '#E2E8F0'},\n"
            "            cbar_kws={'label': 'Correlation', 'shrink': 0.8})\n"
            "ax.set_title('Correlation Matrix', fontsize=15, fontweight='bold', color='#E2E8F0', pad=15)\n"
            "ax.tick_params(colors='#E2E8F0', labelsize=9)\n"
            "plt.tight_layout(pad=1.5)\n"
            "plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')\n"
            "plt.close('all')\n"
            "\n"
            "result = corr"
        ),
    },
    {
        "question": "How many missing values are there in each column?",
        "code": (
            "# Count missing values per column, filter non-zero\n"
            "nulls = df.isnull().sum()\n"
            "nulls = nulls[nulls > 0].sort_values(ascending=False)\n"
            "if len(nulls) == 0:\n"
            "    result = 'No missing values found in any column!'\n"
            "else:\n"
            "    pct = (nulls / len(df) * 100).round(1)\n"
            "    result = pd.DataFrame({'Missing': nulls, '%': pct})"
        ),
    },
    {
        "question": "Show a box plot of salary by department",
        "code": (
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "\n"
            "# Professional box plot\n"
            "palette = ['#8B5CF6','#06D6A0','#F72585','#FFD166','#4CC9F0','#A78BFA']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "fig.patch.set_facecolor('#0f0f1a')\n"
            "ax.set_facecolor('#151528')\n"
            "sns.boxplot(data=df, x='department', y='salary', palette=palette, ax=ax,\n"
            "            saturation=0.85, linewidth=0.8,\n"
            "            flierprops={'markeredgecolor': '#EF476F', 'markersize': 3, 'alpha': 0.6})\n"
            "ax.set_title('Salary Distribution by Department', fontsize=15, fontweight='bold', color='#E2E8F0', pad=15)\n"
            "ax.set_xlabel('Department', fontsize=11, color='#E2E8F0', labelpad=8)\n"
            "ax.set_ylabel('Salary ($)', fontsize=11, color='#E2E8F0', labelpad=8)\n"
            "ax.tick_params(colors='#E2E8F0', labelsize=9)\n"
            "ax.spines['top'].set_visible(False)\n"
            "ax.spines['right'].set_visible(False)\n"
            "ax.spines['left'].set_color('#2D2D44')\n"
            "ax.spines['bottom'].set_color('#2D2D44')\n"
            "ax.yaxis.grid(True, alpha=0.15, color='#4A4A6A', linestyle='--', zorder=0)\n"
            "ax.xaxis.grid(False)\n"
            "plt.xticks(rotation=45, ha='right')\n"
            "plt.tight_layout(pad=1.5)\n"
            "plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')\n"
            "plt.close('all')\n"
            "\n"
            "result = df.groupby('department')['salary'].describe().round(2)"
        ),
    },
]


def format_few_shot_examples() -> str:
    """
    Format few-shot examples into a prompt-ready string.

    Returns:
        A string containing all examples in the expected format.
    """
    lines: list[str] = ["EXAMPLES:"]
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f"\nExample {i}:")
        lines.append(f'Question: "{ex["question"]}"')
        lines.append(f"```python\n{ex['code']}\n```")
    return "\n".join(lines)
