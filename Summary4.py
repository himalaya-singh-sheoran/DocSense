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
                for cat_col in self.column_types['categorical']:
                    context.append(f"{cat_col} '{max_row[cat_col]}'")
                
                self.insights.append(
                    f"Highest {num_col} ({max_val:.1f}) occurred with: " + 
                    ", ".join(context)
                )
    
    def _temporal_analysis(self):
        """Analyze trends over time"""
        if self.column_types['datetime'] and self.column_types['numeric']:
            time_col = self.column_types['datetime'][0]
            
            for num_col in self.column_types['numeric']:
                # Resample to monthly values
                ts = self.df.set_index(time_col)[num_col].resample('M').mean()
                
                if len(ts) > 1:
                    # Calculate trend direction
                    slope, _, _, _, _ = linregress(
                        np.arange(len(ts)), 
                        ts.values
                    )
                    trend = "upward" if slope > 0 else "downward"
                    
                    # Get first and last values
                    first_val = ts.iloc[0]
                    last_val = ts.iloc[-1]
                    change = ((last_val - first_val)/first_val) * 100
                    
                    self.insights.append(
                        f"{num_col} shows {trend} trend over time: "
                        f"from {first_val:.1f} to {last_val:.1f} ({change:.1f}% change)"
                    )
                    
                    # Find best/worst months
                    best_month = ts.idxmax().strftime('%B %Y')
                    worst_month = ts.idxmin().strftime('%B %Y')
                    self.insights.append(
                        f"Peak {num_col} occurred in {best_month}, "
                        f"lowest in {worst_month}"
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
    # Generic test data
    data = {
        'value': [45, 78, 32, 95, 61, 84, 72, 68, 88, 91],
        'category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A'],
        'location': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'],
        'timestamp': pd.date_range('2023-01-01', periods=10, freq='D')
    }
    
    df = pd.DataFrame(data)
    analyzer = GenericDataAnalyzer(df)
    print(analyzer.get_report())
