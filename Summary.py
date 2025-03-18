import pandas as pd
import numpy as np

# Load dataset dynamically
def load_data(file_path):
    """Load CSV or Excel files."""
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

# --- Step 1: Basic Overview ---
def generate_basic_summary(df):
    """Generate general dataset statistics."""
    return {
        "Total Records": df.shape[0],
        "Total Features": df.shape[1],
        "Missing Values": df.isnull().sum().sum(),
        "Duplicate Records": df.duplicated().sum(),
    }

# --- Step 2: Key Relationships Dynamically ---
def analyze_data(df):
    """Extract key insights dynamically based on data types."""
    insights = {}

    # Identify numerical & categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    # Compute insights for numerical columns
    for col in num_cols:
        if df[col].nunique() > 1:  # Avoid single-value columns
            insights[f"Average {col}"] = np.round(df[col].mean(), 2)
            insights[f"Max {col}"] = df[col].max()
            insights[f"Min {col}"] = df[col].min()

    # Find relationships between categorical & numerical columns
    for cat_col in cat_cols:
        if df[cat_col].nunique() > 1:  # Avoid single-value categorical columns
            for num_col in num_cols:
                grouped = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
                if not grouped.empty:
                    insights[f"Highest Avg {num_col} by {cat_col}"] = grouped.idxmax()
                    insights[f"Lowest Avg {num_col} by {cat_col}"] = grouped.idxmin()

    # Identify most frequent categories
    for cat_col in cat_cols:
        if df[cat_col].nunique() > 1:
            insights[f"Most Common {cat_col}"] = df[cat_col].mode()[0]

    return insights

# --- Step 3: Generate Executive Summary ---
def generate_summary_text(summary, insights):
    """Convert extracted insights into a structured executive summary."""
    text = f"""
📊 **Executive Summary**
- The dataset contains {summary['Total Records']} records with {summary['Total Features']} attributes.
- It has {summary['Missing Values']} missing values and {summary['Duplicate Records']} duplicate entries.

🔍 **Key Insights**
"""
    for key, value in insights.items():
        text += f"- {key}: {value}\n"

    return text

# --- Main Execution ---
def main(file_path):
    """Main function to run the executive summary generator."""
    df = load_data(file_path)
    summary = generate_basic_summary(df)
    insights = analyze_data(df)
    summary_text = generate_summary_text(summary, insights)

    print(summary_text)

# Example Usage
file_path = "your_data.csv"  # Change this to your file
main(file_path)
