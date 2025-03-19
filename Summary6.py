import pandas as pd
import numpy as np
from scipy.stats import linregress
from textwrap import wrap

class GenericDataAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.column_types = self._categorize_columns()
        self.insights = []
        
        self._generate_basic_insights()
        self._cross_column_analysis()
        self._temporal_analysis()
        
    def _categorize_columns(self):
        """Classify columns into numeric, categorical, and datetime"""
        col_types = {
            'numeric': [],
            'categorical': [],
            'datetime': []
        }
        
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_types['numeric'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                col_types['datetime'].append(col)
            else:
                col_types['categorical'].append(col)
                
        return col_types
    
    def _generate_basic_insights(self):
        """Generate fundamental data insights"""
        # Basic statistics
        if self.column_types['numeric']:
            for col in self.column_types['numeric']:
                stats = self.df[col].describe()
                self.insights.append(
                    f"The {col} ranges from {stats['min']:.1f} to {stats['max']:.1f} "
                    f"with an average of {stats['mean']:.1f}"
                )
                
        # Category distributions
        if self.column_types['categorical']:
            for col in self.column_types['categorical']:
                top = self.df[col].value_counts().index[0]
                freq = self.df[col].value_counts().iloc[0]/len(self.df)
                self.insights.append(
                    f"Most common {col} is '{top}' ({freq:.1%} of records)"
                )
    
    def _cross_column_analysis(self):
        """Find relationships between numeric and categorical columns"""
        if self.column_types['numeric'] and self.column_types['categorical']:
            for num_col in self.column_types['numeric']:
                # Find maximum value context
                max_val = self.df[num_col].max()
                max_row = self.df[self.df[num_col] == max_val].iloc[0]
                
                context = []
                # Add categorical context
                for cat_col in self.column_types['categorical']:
                    context.append(f"{cat_col} '{max_row[cat_col]}'")
                
                # Add datetime context if available
                if self.column_types['datetime']:
                    for dt_col in self.column_types['datetime']:
                        context.append(f"on {max_row[dt_col].strftime('%Y-%m-%d')}")
                
                self.insights.append(
                    f"Highest {num_col} ({max_val:.1f}) occurred with: " + 
                    ", ".join(context)
                )
    
    def _temporal_analysis(self):
        """Analyze trends over time with date ranges"""
        if self.column_types['datetime'] and self.column_types['numeric']:
            for date_col in self.column_types['datetime']:
                # Get date range for the column
                valid_dates = self.df[date_col].dropna()
                if valid_dates.empty:
                    continue
                
                date_range = f"{valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}"
                
                for num_col in self.column_types['numeric']:
                    # Resample to appropriate frequency
                    time_series = self.df.set_index(date_col)[num_col]
                    if len(time_series) > 90:  # Daily data for >3 months
                        resampled = time_series.resample('M').mean()
                        freq_desc = "monthly"
                    else:
                        resampled = time_series.resample('W').mean()
                        freq_desc = "weekly"
                    
                    if len(resampled) > 1:
                        # Calculate trend
                        slope, _, _, _, _ = linregress(
                            np.arange(len(resampled)), 
                            resampled.values
                        )
                        trend = "upward" if slope > 0 else "downward"
                        
                        # Percentage change
                        change_pct = ((resampled.iloc[-1] - resampled.iloc[0])/resampled.iloc[0] * 100
                        
                        self.insights.append(
                            f"{num_col} shows {trend} trend ({change_pct:+.1f}%) "
                            f"during {date_range} based on {freq_desc} averages"
                        )
                        
                        # Best/worst periods
                        best_period = resampled.idxmax().strftime('%Y-%m-%d')
                        worst_period = resampled.idxmin().strftime('%Y-%m-%d')
                        self.insights.append(
                            f"Peak {num_col} reached {resampled.max():.1f} on {best_period}, "
                            f"lowest {resampled.min():.1f} on {worst_period}"
                        )
    
    def get_report(self, line_width=80):
        """Generate formatted report with wrapped text"""
        report = ["🔍 Data Analysis Insights 🔍", "="*line_width]
        
        for insight in self.insights:
            wrapped = wrap(insight, width=line_width-3)
            report.append("\n  ".join(wrapped))
            
        return "\n".join(report)

# Example Usage
if __name__ == "__main__":
    data = {
        'revenue': [4500, 7800, 3200, 9500, 6100, 8400, 7200, 6800, 8800, 9100],
        'category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A'],
        'region': ['North', 'South', 'North', 'South', 'East', 'West', 'North', 'South', 'East', 'West'],
        'transaction_date': pd.date_range('2023-01-01', periods=10, freq='D')
    }
    
    df = pd.DataFrame(data)
    analyzer = GenericDataAnalyzer(df)
    print(analyzer.get_report())
