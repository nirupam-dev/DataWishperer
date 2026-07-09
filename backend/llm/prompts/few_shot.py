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
            "# Create dark-themed bar chart\n"
            "fig, ax = plt.subplots(figsize=(12, 7))\n"
            "fig.patch.set_facecolor('#1a1a2e')\n"
            "ax.set_facecolor('#16213e')\n"
            "colors = ['#6C5CE7','#00CEC9','#FD79A8','#FDCB6E','#55EFC4','#A29BFE','#FF7675','#74B9FF']\n"
            "ax.bar(range(len(monthly)), monthly.values, color=colors[:len(monthly)], width=0.7)\n"
            "ax.set_xticks(range(len(monthly)))\n"
            "ax.set_xticklabels(monthly.index, rotation=45, ha='right', color='#F0F0F5')\n"
            "ax.set_title('Total Sales by Month', fontsize=18, fontweight='bold', color='#F0F0F5')\n"
            "ax.set_ylabel('Revenue ($)', fontsize=13, color='#F0F0F5')\n"
            "ax.tick_params(colors='#F0F0F5')\n"
            "plt.tight_layout()\n"
            "plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor=plt.gcf().get_facecolor())\n"
            "plt.close()\n"
            "\n"
            "result = f'Monthly sales chart saved. Total: ${monthly.sum():,.2f}'"
        ),
    },
    {
        "question": "Show a histogram of the price distribution",
        "code": (
            "import matplotlib.pyplot as plt\n"
            "\n"
            "# Create dark-themed histogram with stats\n"
            "values = pd.to_numeric(df['price'], errors='coerce').dropna()\n"
            "fig, ax = plt.subplots(figsize=(12, 7))\n"
            "fig.patch.set_facecolor('#1a1a2e')\n"
            "ax.set_facecolor('#16213e')\n"
            "ax.hist(values, bins=30, color='#6C5CE7', edgecolor='white', linewidth=0.5, alpha=0.85)\n"
            "ax.axvline(values.mean(), color='#FF7675', linestyle='--', linewidth=2, label=f'Mean: {values.mean():,.2f}')\n"
            "ax.axvline(values.median(), color='#55EFC4', linestyle='--', linewidth=2, label=f'Median: {values.median():,.2f}')\n"
            "ax.set_title('Price Distribution', fontsize=18, fontweight='bold', color='#F0F0F5')\n"
            "ax.set_xlabel('Price ($)', fontsize=13, color='#F0F0F5')\n"
            "ax.set_ylabel('Frequency', fontsize=13, color='#F0F0F5')\n"
            "ax.tick_params(colors='#F0F0F5')\n"
            "ax.legend(fontsize=10, facecolor='#16213e', edgecolor='none', labelcolor='#F0F0F5')\n"
            "plt.tight_layout()\n"
            "plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor=plt.gcf().get_facecolor())\n"
            "plt.close()\n"
            "\n"
            "result = f'Histogram saved. Mean: ${values.mean():,.2f}, Median: ${values.median():,.2f}, Std: ${values.std():,.2f}'"
        ),
    },
    {
        "question": "Show the correlation matrix for all numeric columns",
        "code": (
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "\n"
            "# Compute correlation matrix with upper triangle mask\n"
            "num_df = df.select_dtypes(include='number')\n"
            "corr = num_df.corr()\n"
            "mask = np.triu(np.ones_like(corr, dtype=bool), k=1)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(14, 10))\n"
            "fig.patch.set_facecolor('#1a1a2e')\n"
            "ax.set_facecolor('#16213e')\n"
            "sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',\n"
            "            vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,\n"
            "            annot_kws={'fontsize': 9, 'color': '#F0F0F5'},\n"
            "            cbar_kws={'label': 'Correlation'})\n"
            "ax.set_title('Correlation Matrix', fontsize=18, fontweight='bold', color='#F0F0F5')\n"
            "ax.tick_params(colors='#F0F0F5')\n"
            "plt.tight_layout()\n"
            "plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor=plt.gcf().get_facecolor())\n"
            "plt.close()\n"
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
            "# Create dark-themed box plot\n"
            "fig, ax = plt.subplots(figsize=(12, 7))\n"
            "fig.patch.set_facecolor('#1a1a2e')\n"
            "ax.set_facecolor('#16213e')\n"
            "palette = ['#6C5CE7','#00CEC9','#FD79A8','#FDCB6E','#55EFC4','#A29BFE']\n"
            "sns.boxplot(data=df, x='department', y='salary', palette=palette, ax=ax,\n"
            "            flierprops={'markeredgecolor': '#FF7675', 'markersize': 4})\n"
            "ax.set_title('Salary Distribution by Department', fontsize=18, fontweight='bold', color='#F0F0F5')\n"
            "ax.set_xlabel('Department', fontsize=13, color='#F0F0F5')\n"
            "ax.set_ylabel('Salary ($)', fontsize=13, color='#F0F0F5')\n"
            "ax.tick_params(colors='#F0F0F5')\n"
            "plt.xticks(rotation=45, ha='right')\n"
            "plt.tight_layout()\n"
            "plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor=plt.gcf().get_facecolor())\n"
            "plt.close()\n"
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
