"""
Few-shot examples for the code generation LLM.

Provides concrete input→output examples that guide the model toward
the expected coding style, output format, and chart aesthetics.

Optimization for Qwen2.5:7B:
    - Fewer examples (4 instead of 6+) to save context tokens
    - Each example is minimal but covers a distinct pattern
    - Chart examples include the exact styling expected
    - Edge case handling shown explicitly
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
            "# Create chart with dark theme\n"
            "plt.style.use('seaborn-v0_8-darkgrid')\n"
            "fig, ax = plt.subplots(figsize=(12, 6))\n"
            "colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(monthly)))\n"
            "ax.bar(range(len(monthly)), monthly.values, color=colors)\n"
            "ax.set_xticks(range(len(monthly)))\n"
            "ax.set_xticklabels(monthly.index, rotation=45, ha='right')\n"
            "ax.set_title('Total Sales by Month', fontsize=16, fontweight='bold')\n"
            "ax.set_xlabel('Month', fontsize=12)\n"
            "ax.set_ylabel('Revenue ($)', fontsize=12)\n"
            "plt.tight_layout()\n"
            "plt.savefig(chart_path, dpi=150, bbox_inches='tight')\n"
            "plt.close()\n"
            "\n"
            "result = f'Monthly sales chart saved. Total: ${monthly.sum():,.2f}'"
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
        "question": "What are the top 5 products by total quantity sold?",
        "code": (
            "# Find top 5 products by total quantity\n"
            "top5 = df.groupby('product')['quantity'].sum().nlargest(5)\n"
            "result = top5.reset_index()\n"
            "result.columns = ['Product', 'Total Quantity']"
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
