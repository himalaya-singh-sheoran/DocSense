import pandas as pd
import numpy as np
from datetime import datetime

def load_data(file_path):
    """Load data from CSV or Excel files with automatic date parsing."""
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, infer_datetime_format=True, parse_dates=True)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path, parse_dates=True)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        return df.infer_objects()
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

def generate_basic_summary(df):
    """Generate comprehensive dataset statistics."""
    missing_values = df.isnull().sum()
    return {
        "Total Records": df.shape[0],
        "Total Features": df.shape[1],
        "Data Types": dict(df.dtypes.value_counts()),
        "Memory Usage (MB)": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
        "Missing Values Total": missing_values.sum(),
        "Top Missing Columns": missing_values.nlargest(3).to_dict(),
        "Duplicate Records": df.duplicated().sum(),
    }

def analyze_data(df, corr_threshold=0.7, max_categories=20):
    """Extract structured insights with statistical analysis."""
    insights = {
        "numerical": {},
        "categorical": {},
        "datetime": {},
        "correlations": [],
        "data_quality": {},
        "relationships": []
    }

    # Column type identification
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    # Numerical analysis
    for col in num_cols:
        if df[col].nunique() == 0:
            continue
        stats = df[col].describe(percentiles=[.25, .5, .75]).to_dict()
        stats.update({
            "skewness": round(df[col].skew(), 2),
            "kurtosis": round(df[col].kurtosis(), 2),
            "outliers": detect_outliers(df[col])
        })
        insights["numerical"][col] = {k: round(v, 2) if isinstance(v, float) else v 
                                    for k, v in stats.items()}

    # Categorical analysis
    for col in cat_cols:
        unique_count = df[col].nunique()
        if unique_count == 0:
            continue
        value_counts = df[col].value_counts().nlargest(5)
        insights["categorical"][col] = {
            "unique_count": unique_count,
            "top_values": (value_counts / len(df)).round(3).to_dict(),
            "high_cardinality": unique_count > max_categories
        }

    # Time series analysis
    for col in datetime_cols:
        if num_cols:
            time_insights = analyze_time_series(df, col, num_cols[0])
            insights["datetime"][col] = time_insights

    # Correlation analysis
    if len(num_cols) > 1:
        insights["correlations"] = find_strong_correlations(df[num_cols], corr_threshold)

    # Data quality issues
    insights["data_quality"] = {
        col: f"{missing_pct}% missing" 
        for col, missing_pct in (df.isnull().mean() * 100).round(1).items()
        if missing_pct > 5
    }

    # Relationships analysis
    insights["relationships"] = find_significant_relationships(df, num_cols, cat_cols, max_categories)
    
    return insights

def detect_outliers(series):
    """Identify outliers using IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    bounds = (q1 - 1.5*iqr, q3 + 1.5*iqr)
    return {
        "count": series[(series < bounds[0]) | (series > bounds[1])].count(),
        "percentage": round((series[(series < bounds[0]) | (series > bounds[1])).count() / len(series) * 100, 1)
    }

def analyze_time_series(df, date_col, value_col):
    """Analyze monthly trends for time series data."""
    df = df.copy()
    df['_period'] = df[date_col].dt.to_period('M')
    monthly_stats = df.groupby('_period')[value_col].agg(['mean', 'sum']).reset_index()
    monthly_stats['_period'] = monthly_stats['_period'].astype(str)
    return {
        "monthly_mean": monthly_stats.set_index('_period')['mean'].to_dict(),
        "monthly_total": monthly_stats.set_index('_period')['sum'].to_dict()
    }

def find_strong_correlations(df, threshold):
    """Identify strongly correlated numerical features."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    return [
        {"features": f"{i} & {j}", "value": round(corr_matrix.loc[i, j], 2)}
        for i, j in zip(*np.where(upper > threshold))
    ]

def find_significant_relationships(df, num_cols, cat_cols, max_categories):
    """Find significant categorical-numerical relationships."""
    relationships = []
    for cat_col in cat_cols:
        if 1 < df[cat_col].nunique() <= max_categories:
            for num_col in num_cols:
                grouped = df.groupby(cat_col)[num_col].mean()
                overall_mean = df[num_col].mean()
                max_val = grouped.max()
                max_cat = grouped.idxmax()
                pct_diff = ((max_val - overall_mean) / overall_mean) * 100
                if abs(pct_diff) > 20:  # Only report significant differences
                    relationships.append({
                        "category": cat_col,
                        "metric": num_col,
                        "highest_group": {
                            "name": max_cat,
                            "value": round(max_val, 2),
                            "pct_diff": round(pct_diff, 1)
                        }
                    })
    return relationships

def generate_summary_text(summary, insights):
    """Generate structured business-friendly report."""
    report = [
        "📊 Executive Summary",
        f"- Dataset contains {summary['Total Records']} records with {summary['Total Features']} features",
        f"- Memory Usage: {summary['Memory Usage (MB)']} MB",
        f"- Missing Values: {summary['Missing Values Total']} (Top: {', '.join([f'{k}: {v}' for k, v in summary['Top Missing Columns'].items()])})",
        f"- Duplicates: {summary['Duplicate Records']}",
        "\n🔍 Critical Data Quality Issues:"
    ]
    
    # Data quality issues
    report.extend([f"- {col}: {msg}" for col, msg in insights['data_quality'].items()])
    
    # Numerical insights
    report.append("\n📈 Key Numerical Insights:")
    for col, stats in insights['numerical'].items():
        report.append(
            f"- {col}: Avg {stats['mean']} (25% {stats['25%']}, 75% {stats['75%']}) | "
            f"Outliers: {stats['outliers']['count']} ({stats['outliers']['percentage']}%)"
        )
    
    # Categorical insights
    report.append("\n📊 Categorical Distributions:")
    for col, data in insights['categorical'].items():
        if data['high_cardinality']:
            report.append(f"- {col}: High cardinality ({data['unique_count']} categories)")
        else:
            top_values = [f"{k} ({v*100:.1f}%)" for k, v in data['top_values'].items()]
            report.append(f"- {col}: Top values - {', '.join(top_values)}")
    
    # Time series insights
    if insights['datetime']:
        report.append("\n📅 Time Series Trends:")
        for col, data in insights['datetime'].items():
            last_period = max(data['monthly_mean'].keys())
            report.append(
                f"- {col}: Last {last_period} - Mean: {data['monthly_mean'][last_period]:.2f}, "
                f"Total: {data['monthly_total'][last_period]:.2f}"
            )
    
    # Correlations
    if insights['correlations']:
        report.append("\n🔗 Strong Correlations:")
        report.extend([f"- {x['features']}: {x['value']}" for x in insights['correlations']])
    
    # Relationships
    if insights['relationships']:
        report.append("\n🤝 Significant Relationships:")
        for rel in insights['relationships']:
            report.append(
                f"- {rel['category']} has highest {rel['metric']} in {rel['highest_group']['name']} "
                f"({rel['highest_group']['value']}, {rel['highest_group']['pct_diff']}% above average)"
            )
    
    return "\n".join(report)

def main(file_path):
    """Generate comprehensive business insights report."""
    df = load_data(file_path)
    summary = generate_basic_summary(df)
    insights = analyze_data(df)
    report = generate_summary_text(summary, insights)
    
    # Save and print report
    with open("business_insights_report.txt", "w") as f:
        f.write(report)
    print(report)

# Example usage
if __name__ == "__main__":
    file_path = "your_data.csv"  # Change to your dataset
    main(file_path)
