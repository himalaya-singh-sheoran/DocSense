import pandas as pd
import numpy as np
import warnings
from datetime import datetime

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
                # Attempt datetime conversion with warning suppression
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        # Try inferring datetime with mixed format support
                        self.df[col] = pd.to_datetime(
                            self.df[col], 
                            infer_datetime_format=True,
                            errors='coerce'
                        )
                        # Only consider as datetime if at least 50% values converted
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
        
        # Dataset overview
        summary.append(f"Dataset contains {len(self.df)} records with {len(self.df.columns)} columns")
        
        # Numeric columns summary
        if self.column_types['numeric']:
            summary.append("\nNumerical Values Analysis:")
            for col in self.column_types['numeric']:
                stats = self.df[col].describe()
                summary.append(
                    f"- {col}: "
                    f"Average {stats['mean']:.1f}, "
                    f"ranges from {stats['min']:.1f} to {stats['max']:.1f}"
                )
        
        # Categorical columns summary
        if self.column_types['categorical']:
            summary.append("\nCategory Analysis:")
            for col in self.column_types['categorical']:
                top_value = self.df[col].mode()[0]
                top_percent = (self.df[col].value_counts().max() / len(self.df)) * 100
                summary.append(
                    f"- {col}: "
                    f"Most common value '{top_value}' ({top_percent:.1f}% of records)"
                )
        
        # Datetime columns summary
        if self.column_types['datetime']:
            summary.append("\nDate Analysis:")
            for col in self.column_types['datetime']:
                date_range = f"{self.df[col].min().strftime('%Y-%m-%d')} to {self.df[col].max().strftime('%Y-%m-%d')}"
                summary.append(f"- {col} ranges from {date_range}")
        
        return "\n".join(summary)

    def _generate_insights(self):
        """Generate business insights based on data patterns"""
        insights = []
        
        # Missing values insight
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            insights.append("⚠️ Data Quality Note: Found missing values in columns - " + 
                          ", ".join([f"{col} ({count})" for col, count in missing_values[missing_values > 0].items()]))
        
        # Numeric correlations
        if len(self.column_types['numeric']) > 1:
            corr_matrix = self.df[self.column_types['numeric']].corr().abs()
            high_corr = corr_matrix[(corr_matrix > 0.7) & (corr_matrix < 1)]
            if not high_corr.stack().empty:
                strongest = high_corr.stack().idxmax()
                insights.append(f"📈 Strong relationship between {strongest[0]} and {strongest[1]} (consider analyzing together)")
        
        # Category-numeric relationships
        if self.column_types['categorical'] and self.column_types['numeric']:
            for cat_col in self.column_types['categorical']:
                for num_col in self.column_types['numeric']:
                    grouped = self.df.groupby(cat_col)[num_col].mean()
                    # Avoid division by zero and ensure meaningful ratio
                    if grouped.min() > 0 and (grouped.max() / grouped.min() > 2):
                        insights.append(
                            f"🔍 {cat_col} significantly affects {num_col}: "
                            f"{grouped.idxmax()} has {grouped.max():.1f} vs {grouped.idxmin()} with {grouped.min():.1f}"
                        )
        
        # Time-based trends
        for date_col in self.column_types['datetime']:
            if self.column_types['numeric']:
                try:
                    # Set the datetime column as index and resample monthly on the first numeric column
                    time_series = self.df.set_index(date_col)[self.column_types['numeric'][0]].resample('M').mean()
                    pct_change = time_series.pct_change().mean()
                    if pct_change > 0.1:
                        insights.append(f"📅 Clear upward trend in {self.column_types['numeric'][0]} over time")
                    elif pct_change < -0.1:
                        insights.append(f"📅 Downward trend observed in {self.column_types['numeric'][0]} over time")
                except Exception as e:
                    # If resampling fails, skip this date column.
                    continue
        
        return insights

    def show_report(self):
        """Generate final formatted report"""
        report = [
            "📊 Data Overview:",
            self.summary,
            "\n🔑 Key Insights:",
            *self.insights
        ]
        return "\n".join(report)

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        'Sales': [100, 150, 200, 180, 300],
        'Region': ['North', 'North', 'South', 'South', 'West'],
        'Date': pd.date_range(start='2023-01-01', periods=5),
        'Customer Rating': [4.5, 4.8, 3.9, 4.2, 4.7]
    }
    
    df = pd.DataFrame(data)
    analyzer = DataFrameInsightGenerator(df)
    print(analyzer.show_report())
