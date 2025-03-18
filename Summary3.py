import pandas as pd
import numpy as np
import warnings
from scipy import stats

class DataFrameInsightGenerator:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.column_types = self._identify_column_types()
        self.summary = self._generate_summary()
        self.insights = self._generate_insights()

    def _identify_column_types(self):
        """Classify columns into numeric, categorical, and datetime"""
        column_types = {
            'numeric': [],
            'categorical': [],
            'datetime': []
        }
        
        for col in self.df.columns:
            if np.issubdtype(self.df[col].dtype, np.number):
                column_types['numeric'].append(col)
            elif self.df[col].dtype == 'object':
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        self.df[col] = pd.to_datetime(
                            self.df[col], 
                            infer_datetime_format=True,
                            errors='coerce'
                        )
                        if self.df[col].notna().mean() > 0.5:
                            column_types['datetime'].append(col)
                        else:
                            column_types['categorical'].append(col)
                    except:
                        column_types['categorical'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                column_types['datetime'].append(col)
            else:
                column_types['categorical'].append(col)
                
        return column_types

    def _generate_summary(self):
        """Generate basic summary statistics"""
        summary = []
        
        summary.append(f"📄 Dataset contains {len(self.df):,} records with {len(self.df.columns)} columns")
        
        if self.column_types['numeric']:
            summary.append("\n🔢 Numerical Analysis:")
            for col in self.column_types['numeric']:
                stats = self.df[col].describe()
                summary.append(
                    f"- {col}: "
                    f"Avg: {stats['mean']:.1f} | "
                    f"Range: {stats['min']:.1f}-{stats['max']:.1f} | "
                    f"25%: {stats['25%']:.1f} | 75%: {stats['75%']:.1f}"
                )
        
        if self.column_types['categorical']:
            summary.append("\n🏷️ Category Analysis:")
            for col in self.column_types['categorical']:
                mode_data = self.df[col].mode()
                if not mode_data.empty:
                    top_value = mode_data.iloc[0]
                    top_percent = (self.df[col].value_counts().iloc[0] / len(self.df)) * 100
                    unique_count = self.df[col].nunique()
                    summary.append(
                        f"- {col}: "
                        f"Top: '{top_value}' ({top_percent:.1f}%) | "
                        f"Unique values: {unique_count}"
                    )
        
        if self.column_types['datetime']:
            summary.append("\n📅 Date Analysis:")
            for col in self.column_types['datetime']:
                date_range = f"{self.df[col].min().strftime('%d %b %Y')} to {self.df[col].max().strftime('%d %b %Y')}"
                summary.append(f"- {col} spans {date_range}")
        
        return "\n".join(summary)

    def _generate_insights(self):
        """Generate actionable business insights"""
        insights = []
        
        # Correlation insights
        if len(self.column_types['numeric']) > 1:
            corr_matrix = self.df[self.column_types['numeric']].corr().abs()
            for (col1, col2), value in corr_matrix.stack().items():
                if col1 != col2 and value > 0.7:
                    insights.append(
                        f"📈 Strong positive relationship between {col1} and {col2} "
                        f"(correlation: {value:.2f})"
                    )
                elif col1 != col2 and value < -0.7:
                    insights.append(
                        f"📉 Strong negative relationship between {col1} and {col2} "
                        f"(correlation: {value:.2f})"
                    )
        
        # Category performance analysis
        if self.column_types['categorical'] and self.column_types['numeric']:
            for cat_col in self.column_types['categorical']:
                for num_col in self.column_types['numeric']:
                    category_stats = self.df.groupby(cat_col)[num_col].agg(['mean', 'count'])
                    if not category_stats.empty:
                        top_category = category_stats['mean'].idxmax()
                        bottom_category = category_stats['mean'].idxmin()
                        insights.append(
                            f"🏆 {cat_col} performance: "
                            f"Highest {num_col} in {top_category} "
                            f"({category_stats['mean'][top_category]:.1f}), "
                            f"Lowest in {bottom_category} "
                            f"({category_stats['mean'][bottom_category]:.1f})"
                        )
        
        # Time series trends
        for date_col in self.column_types['datetime']:
            if self.column_types['numeric']:
                num_col = self.column_types['numeric'][0]
                monthly_trend = self.df.set_index(date_col)[num_col].resample('M').mean()
                if len(monthly_trend) > 1:
                    latest_growth = (monthly_trend[-1] - monthly_trend[-2])/monthly_trend[-2] * 100
                    insights.append(
                        f"⏳ Recent trend: {num_col} changed by {latest_growth:.1f}% "
                        f"from {monthly_trend.index[-2].strftime('%b')} to "
                        f"{monthly_trend.index[-1].strftime('%b')}"
                    )
        
        # Outlier detection
        for col in self.column_types['numeric']:
            z_scores = np.abs(stats.zscore(self.df[col]))
            outliers = self.df[z_scores > 3]
            if not outliers.empty:
                insights.append(
                    f"⚠️ Potential outliers detected in {col} "
                    f"(values above {outliers[col].max():.1f} or "
                    f"below {outliers[col].min():.1f})"
                )
        
        # Distribution analysis
        for col in self.column_types['numeric']:
            skewness = self.df[col].skew()
            if abs(skewness) > 1:
                insights.append(
                    f"📊 {col} distribution is skewed "
                    f"{'right' if skewness > 0 else 'left'} "
                    f"(most values clustered on the {'lower' if skewness > 0 else 'higher'} side)"
                )
        
        return insights

    def show_report(self):
        """Generate final formatted report"""
        report = [
            "📊 Data Overview:",
            self.summary,
            "\n🔍 Key Insights:",
            *self.insights
        ]
        return "\n".join(report)

# Example usage
if __name__ == "__main__":
    data = {
        'Sales': [120, 150, 180, 200, 250, 300, 280, 220, 190, 210],
        'Region': ['North', 'North', 'South', 'South', 'West', 'West', 'East', 'East', 'North', 'South'],
        'Date': pd.date_range(start='2023-01-01', periods=10, freq='M'),
        'Profit': [15, 18, 22, 25, 30, 35, 28, 24, 20, 22]
    }
    
    df = pd.DataFrame(data)
    analyzer = DataFrameInsightGenerator(df)
    print(analyzer.show_report())
