"""
Few-shot examples for the code generation LLM.

Provides concrete input→output examples that guide the model toward
the expected coding style, output format, and chart aesthetics.

Optimization for Qwen2.5:7B:
    - Fewer examples (6 instead of 10+) to save context tokens
    - Each example covers a distinct pattern
    - Chart examples include premium dark dashboard styling
    - Covers bar, histogram, scatter, correlation, box, donut patterns
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
            "# Premium dark-themed bar chart\n"
            "palette = ['#8B5CF6','#06D6A0','#F72585','#FFD166','#4CC9F0','#A78BFA','#EF476F','#118AB2']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "fig.patch.set_facecolor('#0E0E1A')\n"
            "ax.set_facecolor('#131325')\n"
            "\n"
            "# Subtle glow layer behind bars\n"
            "colors = [palette[i % len(palette)] for i in range(len(monthly))]\n"
            "ax.bar(range(len(monthly)), monthly.values, width=0.7,\n"
            "       color=[c + '18' for c in colors], zorder=1)\n"
            "\n"
            "# Main bars\n"
            "bars = ax.bar(range(len(monthly)), monthly.values,\n"
            "              color=colors, width=0.55, edgecolor='none',\n"
            "              alpha=0.92, zorder=3)\n"
            "\n"
            "# Value labels on top\n"
            "max_val = max(monthly.values)\n"
            "for bar, val in zip(bars, monthly.values):\n"
            "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val*0.02,\n"
            "            f'${val:,.0f}', ha='center', va='bottom', fontsize=10,\n"
            "            fontweight='bold', color='#FFFFFF')\n"
            "\n"
            "ax.set_xticks(range(len(monthly)))\n"
            "ax.set_xticklabels(monthly.index, rotation=45, ha='right', fontsize=10, color='#8892A0')\n"
            "ax.set_title('Total Sales by Month', fontsize=16, fontweight='bold', color='#FFFFFF', pad=18)\n"
            "ax.set_ylabel('Revenue ($)', fontsize=12, color='#B0BEC5', labelpad=10)\n"
            "ax.tick_params(colors='#8892A0', labelsize=10)\n"
            "for s in ax.spines.values(): s.set_visible(False)\n"
            "ax.yaxis.grid(True, alpha=0.08, color='#3A3A5C', linestyle='-')\n"
            "ax.xaxis.grid(False)\n"
            "plt.tight_layout(pad=2.0)\n"
            "plt.savefig(chart_path, dpi=250, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')\n"
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
            "# Premium histogram with statistical annotations\n"
            "values = pd.to_numeric(df['price'], errors='coerce').dropna()\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "fig.patch.set_facecolor('#0E0E1A')\n"
            "ax.set_facecolor('#131325')\n"
            "n, bins, patches = ax.hist(values, bins=30, color='#8B5CF6', edgecolor='#0E0E1A',\n"
            "                            linewidth=0.8, alpha=0.88, zorder=3)\n"
            "\n"
            "# Gradient coloring on histogram bins\n"
            "import matplotlib.cm as cm\n"
            "norm = plt.Normalize(n.min(), n.max())\n"
            "for count, patch in zip(n, patches):\n"
            "    color = cm.plasma(norm(count))\n"
            "    patch.set_facecolor(color)\n"
            "    patch.set_alpha(0.88)\n"
            "\n"
            "# Statistical lines\n"
            "ax.axvline(values.mean(), color='#F72585', linestyle='--', linewidth=2, alpha=0.9,\n"
            "           label=f'Mean: ${values.mean():,.2f}', zorder=4)\n"
            "ax.axvline(values.median(), color='#06D6A0', linestyle='--', linewidth=2, alpha=0.9,\n"
            "           label=f'Median: ${values.median():,.2f}', zorder=4)\n"
            "\n"
            "ax.set_title('Price Distribution', fontsize=16, fontweight='bold', color='#FFFFFF', pad=18)\n"
            "ax.set_xlabel('Price ($)', fontsize=12, color='#B0BEC5', labelpad=10)\n"
            "ax.set_ylabel('Frequency', fontsize=12, color='#B0BEC5', labelpad=10)\n"
            "ax.tick_params(colors='#8892A0', labelsize=10)\n"
            "for s in ax.spines.values(): s.set_visible(False)\n"
            "ax.yaxis.grid(True, alpha=0.08, color='#3A3A5C', linestyle='-')\n"
            "ax.xaxis.grid(False)\n"
            "ax.legend(fontsize=10, facecolor='#1A1A2E', edgecolor='none', labelcolor='#E2E8F0', framealpha=0.85)\n"
            "plt.tight_layout(pad=2.0)\n"
            "plt.savefig(chart_path, dpi=250, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')\n"
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
            "# Premium correlation heatmap with masked upper triangle\n"
            "num_df = df.select_dtypes(include='number')\n"
            "corr = num_df.corr()\n"
            "mask = np.triu(np.ones_like(corr, dtype=bool), k=1)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8, 6))\n"
            "fig.patch.set_facecolor('#0E0E1A')\n"
            "ax.set_facecolor('#131325')\n"
            "sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='magma',\n"
            "            vmin=-1, vmax=1, center=0, square=True, linewidths=1,\n"
            "            linecolor='#0E0E1A',\n"
            "            annot_kws={'fontsize': 10, 'color': '#FFFFFF', 'fontweight': 'bold'},\n"
            "            cbar_kws={'label': 'Correlation', 'shrink': 0.8})\n"
            "ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold', color='#FFFFFF', pad=18)\n"
            "ax.tick_params(colors='#8892A0', labelsize=10)\n"
            "plt.tight_layout(pad=2.0)\n"
            "plt.savefig(chart_path, dpi=250, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')\n"
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
            "# Premium box plot with refined aesthetics\n"
            "palette = ['#8B5CF6','#06D6A0','#F72585','#FFD166','#4CC9F0','#A78BFA']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "fig.patch.set_facecolor('#0E0E1A')\n"
            "ax.set_facecolor('#131325')\n"
            "sns.boxplot(data=df, x='department', y='salary', palette=palette, ax=ax,\n"
            "            saturation=0.85, linewidth=0.8, width=0.55,\n"
            "            flierprops={'markeredgecolor': '#F72585', 'markersize': 4, 'alpha': 0.6},\n"
            "            medianprops={'color': '#FFD166', 'linewidth': 2.5},\n"
            "            whiskerprops={'color': '#8892A0', 'linewidth': 1.2},\n"
            "            capprops={'color': '#8892A0', 'linewidth': 1.2},\n"
            "            boxprops={'edgecolor': 'none'})\n"
            "ax.set_title('Salary Distribution by Department', fontsize=16, fontweight='bold', color='#FFFFFF', pad=18)\n"
            "ax.set_xlabel('Department', fontsize=12, color='#B0BEC5', labelpad=10)\n"
            "ax.set_ylabel('Salary ($)', fontsize=12, color='#B0BEC5', labelpad=10)\n"
            "ax.tick_params(colors='#8892A0', labelsize=10)\n"
            "for s in ax.spines.values(): s.set_visible(False)\n"
            "ax.yaxis.grid(True, alpha=0.08, color='#3A3A5C', linestyle='-')\n"
            "ax.xaxis.grid(False)\n"
            "plt.xticks(rotation=45, ha='right')\n"
            "plt.tight_layout(pad=2.0)\n"
            "plt.savefig(chart_path, dpi=250, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')\n"
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
