import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

class SmartPlotGenerator:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.generated_files = []
        self.generated_figures = []
        self._prepare_data()
        self.debug_info = {}  # Store debugging information

    def _prepare_data(self):
        """Enhanced data preparation with better type detection"""
        # Convert object columns to proper types
        for col in self.df.select_dtypes(include=['object']):
            try:
                self.df[col] = pd.to_datetime(self.df[col])
                print(f"Converted column {col} to datetime")
            except (TypeError, ValueError):
                try:
                    self.df[col] = pd.to_numeric(self.df[col])
                    print(f"Converted column {col} to numeric")
                except:
                    pass

        # Remove columns with all null values
        self.df = self.df.dropna(axis=1, how='all')

    def _get_column_type(self, col: str) -> str:
        """Enhanced column typing with cardinality checks"""
        if pd.api.types.is_datetime64_any_dtype(self.df[col]):
            return 'date'
        elif pd.api.types.is_numeric_dtype(self.df[col]):
            return 'numeric'
        else:
            nunique = self.df[col].nunique()
            if nunique < 20 and nunique < 0.5 * len(self.df):
                return 'categorical'
            return 'high_cardinality'

    def generate_insights(self, output_dir: str = ".") -> List[go.Figure]:
        """Generate insights with built-in debug checks"""
        self.output_dir = output_dir
        self._analyze_columns()
        
        # Store debug information
        self.debug_info = {
            'column_types': self.column_types,
            'date_columns': self.date_columns,
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns
        }
        
        print("\n=== Data Profile ===")
        print(f"Date columns: {self.date_columns}")
        print(f"Numeric columns: {self.numeric_columns}")
        print(f"Categorical columns: {self.categorical_columns}")
        
        if not self.numeric_columns:
            print("\n⚠️ Warning: No numeric columns found - cannot generate insights")
            return []

        self._create_time_series_plots()
        self._create_categorical_plots()
        self._create_numeric_distributions()
        
        print(f"\n✅ Generated {len(self.generated_figures)} insights")
        return self.generated_figures

    def _analyze_columns(self):
        """Analyze columns with enhanced validation"""
        self.column_types = {col: self._get_column_type(col) for col in self.df.columns}
        
        self.date_columns = [col for col, t in self.column_types.items() if t == 'date']
        self.numeric_columns = [col for col, t in self.column_types.items() if t == 'numeric']
        self.categorical_columns = [col for col, t in self.column_types.items() 
                                   if t == 'categorical' and col not in self.date_columns]

    def _create_time_series_plots(self):
        """Create time series plots with fallback to index"""
        if self.date_columns:
            for date_col in self.date_columns:
                self._create_time_series_chart(date_col)
        else:
            print("⚠️ No date columns found - using index for time series")
            self.df['index'] = self.df.index
            self._create_time_series_chart('index')

    def _create_time_series_chart(self, date_col: str):
        """Helper to create time series chart"""
        numeric_cols = [col for col in self.numeric_columns if col != date_col]
        if not numeric_cols:
            print(f"⚠️ No numeric columns to plot against {date_col}")
            return

        fig = go.Figure()
        for num_col in numeric_cols:
            fig.add_trace(go.Scatter(
                x=self.df[date_col],
                y=self.df[num_col],
                name=num_col,
                mode='lines+markers'
            ))

        fig.update_layout(
            title=f"Time Series: {', '.join(numeric_cols)}",
            xaxis_title=date_col,
            yaxis_title="Value",
            template='plotly_white'
        )
        self._save_chart(fig, f"timeseries_{date_col}.html")

    def _create_categorical_plots(self):
        """Create categorical plots with enhanced grouping"""
        for cat_col in self.categorical_columns:
            if not self.numeric_columns:
                continue

            agg_df = self.df.groupby(cat_col)[self.numeric_columns].mean().reset_index()
            
            fig = go.Figure()
            for num_col in self.numeric_columns:
                fig.add_trace(go.Bar(
                    x=agg_df[cat_col],
                    y=agg_df[num_col],
                    name=num_col
                ))

            fig.update_layout(
                title=f"{cat_col} Analysis",
                xaxis_title=cat_col,
                template='plotly_white'
            )
            self._save_chart(fig, f"categorical_{cat_col}.html")

    def _create_numeric_distributions(self):
        """Create distributions for all numeric columns"""
        for num_col in self.numeric_columns:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=self.df[num_col], name=num_col))
            fig.update_layout(
                title=f"{num_col} Distribution",
                template='plotly_white'
            )
            self._save_chart(fig, f"distribution_{num_col}.html")

    def _save_chart(self, fig: go.Figure, filename: str):
        """Save chart and track outputs"""
        path = Path(self.output_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))
        self.generated_files.append(str(path))
        self.generated_figures.append(fig)

# Test with sample data
if __name__ == "__main__":
    # Create test data with clear patterns
    test_df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=50),
        'sales': np.random.randint(100, 500, 50).cumsum(),
        'region': np.random.choice(['North', 'South', 'East'], 50),
        'temperature': np.random.normal(20, 5, 50)
    })
    
    print("=== Test Dataset ===")
    print(test_df.head())
    
    analyzer = SmartPlotGenerator(test_df)
    insights = analyzer.generate_insights()
    
    print("\n=== Generated Files ===")
    print("\n".join(analyzer.generated_files))
