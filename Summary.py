import pandas as pd
import numpy as np

class ExecutiveSummaryGenerator:
    def __init__(self, file_path):
        """
        Initialize the summary generator with the file path.
        Loads the data and performs any initial processing.
        """
        self.file_path = file_path
        self.df = self.load_data()
        self.basic_summary = self.generate_basic_summary()
        self.insights = self.analyze_managerial_insights()

    def load_data(self):
        """Load CSV or Excel files based on the file extension."""
        if self.file_path.endswith(".csv"):
            df = pd.read_csv(self.file_path)
        elif self.file_path.endswith(".xlsx"):
            df = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel.")
        return df

    def generate_basic_summary(self):
        """Generate basic statistics of the dataset."""
        df = self.df
        summary = {
            "Total Records": df.shape[0],
            "Total Attributes": df.shape[1],
            "Total Missing Values": df.isnull().sum().sum(),
            "Duplicate Records": df.duplicated().sum(),
        }
        return summary

    def analyze_managerial_insights(self):
        """
        Extract plain-language insights for higher management.
        This includes overall averages, comparisons across groups,
        data quality issues, and time series trends (if date/time data is available).
        """
        df = self.df.copy()  # Work on a copy
        insights = {}

        # Identify numeric and categorical columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Identify datetime columns: either already datetime or convertible
        dt_cols = []
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.datetime64):
                dt_cols.append(col)
            else:
                try:
                    converted = pd.to_datetime(df[col], errors='raise')
                    if converted.notnull().mean() > 0.8:  # At least 80% conversion success
                        df[col] = converted
                        dt_cols.append(col)
                except Exception:
                    continue

        # 1. Overall Numeric Metrics
        if num_cols:
            for col in num_cols:
                overall_avg = np.round(df[col].mean(), 2)
                insights[f"Overall average {col}"] = f"The overall average {col} is {overall_avg}."

        # 2. Group Comparisons: Relationship between categorical and numeric data
        for cat in cat_cols:
            if df[cat].nunique() <= 1:
                continue
            for num in num_cols:
                grouped = df.groupby(cat)[num].mean()
                if grouped.empty or grouped.nunique() == 1:
                    continue
                best_group = grouped.idxmax()
                best_value = np.round(grouped.max(), 2)
                worst_group = grouped.idxmin()
                worst_value = np.round(grouped.min(), 2)
                overall = np.round(df[num].mean(), 2)

                insight_text = (
                    f"For {num}, the overall average is {overall}. The group '{best_group}' has the highest average at {best_value}, "
                    f"while the group '{worst_group}' has the lowest at {worst_value}."
                )
                insights[f"{num} by {cat}"] = insight_text

        # 3. Data Completeness: Flag columns with high missing data (> 20%)
        missing_details = {}
        for col in df.columns:
            missing_pct = np.round(df[col].isnull().mean() * 100, 2)
            if missing_pct > 20:
                missing_details[col] = f"{missing_pct}% missing"
        if missing_details:
            insights["Data Completeness"] = (
                "Some attributes have significant missing data: " +
                ", ".join([f"{k} ({v})" for k, v in missing_details.items()])
            )
        else:
            insights["Data Completeness"] = "All attributes have good data coverage."

        # 4. Time Series Insights (if any datetime columns exist)
        time_series_insights = {}
        if dt_cols and num_cols:
            for dt in dt_cols:
                start_date = df[dt].min().date()
                end_date = df[dt].max().date()
                time_series_insights[dt] = f"Data in '{dt}' spans from {start_date} to {end_date}."
                # As an example, group by month using the first numeric column
                numeric_example = num_cols[0]
                df['__month__'] = df[dt].dt.to_period('M')
                trend = df.groupby('__month__')[numeric_example].mean().reset_index()
                if not trend.empty:
                    best_idx = trend[numeric_example].idxmax()
                    worst_idx = trend[numeric_example].idxmin()
                    best_month = trend.loc[best_idx, '__month__']
                    best_value = np.round(trend.loc[best_idx, numeric_example], 2)
                    worst_month = trend.loc[worst_idx, '__month__']
                    worst_value = np.round(trend.loc[worst_idx, numeric_example], 2)
                    time_series_insights[f"{dt} Trend for {numeric_example}"] = (
                        f"For {numeric_example} based on '{dt}', the highest monthly average was in {best_month} at {best_value}, "
                        f"and the lowest in {worst_month} at {worst_value}."
                    )
                df.drop(columns='__month__', inplace=True)
        if time_series_insights:
            insights["Time Series Insights"] = time_series_insights

        # 5. Additional General Observations
        insights["General Observations"] = (
            "The insights above highlight key performance metrics and differences across groups, "
            "as well as data quality and time-based trends that may require attention."
        )

        return insights

    def generate_management_summary(self):
        """
        Combine the basic summary and insights into a plain-language executive summary.
        This summary is tailored for higher management.
        """
        basic = self.basic_summary
        insights = self.insights

        summary_text = "\nExecutive Summary for Management:\n"
        summary_text += "-------------------------------------\n"
        summary_text += f"- Total Records: {basic['Total Records']}\n"
        summary_text += f"- Total Attributes: {basic['Total Attributes']}\n"
        summary_text += f"- Missing Values: {basic['Total Missing Values']}\n"
        summary_text += f"- Duplicate Records: {basic['Duplicate Records']}\n\n"

        summary_text += "Key Insights:\n"
        for key, value in insights.items():
            if isinstance(value, dict):
                summary_text += f"\n* {key}:\n"
                for sub_key, sub_value in value.items():
                    summary_text += f"    - {sub_value}\n"
            else:
                summary_text += f"- {value}\n"

        summary_text += "\nThese insights provide a concise yet comprehensive view of performance, trends, and data quality."
        return summary_text

    def run(self):
        """Generate and print the executive summary."""
        management_summary = self.generate_management_summary()
        print(management_summary)


# -----------------------------
# Example Usage:
# -----------------------------
if __name__ == "__main__":
    # Replace 'your_data.csv' with the path to your data file (CSV or XLSX)
    generator = ExecutiveSummaryGenerator("your_data.csv")
    generator.run()
