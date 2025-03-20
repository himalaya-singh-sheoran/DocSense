import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

class SmartPlotGenerator:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.generated_files = []
        self.generated_figures = []
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Convert potential date columns to datetime format"""
        for col in self.df.columns:
            try:
                self.df[col] = pd.to_datetime(self.df[col])
            except (TypeError, ValueError):
                pass

    def _save_chart(self, fig: go.Figure, filename: str) -> None:
        """Save chart and track generated files"""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))
        self.generated_files.append(str(path.absolute())

    def _get_column_type(self, col: str) -> str:
        """Classify columns into date, numeric, or categorical"""
        if pd.api.types.is_datetime64_any_dtype(self.df[col]):
            return 'date'
        elif pd.api.types.is_numeric_dtype(self.df[col]):
            return 'numeric'
        else:
            return 'categorical' if self.df[col].nunique() < 20 else 'high_cardinality'

    def generate_insights(self, output_dir: str = "plots") -> List[go.Figure]:
        """Main method to generate all smart visualizations"""
        self.output_dir = output_dir
        self._analyze_columns()
        
        # Generate time series plots
        if self.date_columns:
            self._create_time_series_plots()
        
        # Generate categorical-numeric plots
        if self.categorical_columns and self.numeric_columns:
            self._create_categorical_plots()
        
        # Generate numeric distribution plots
        if self.numeric_columns:
            self._create_numeric_distributions()
        
        return self.generated_figures

    def _analyze_columns(self) -> None:
        """Analyze and classify dataframe columns"""
        self.column_types = {col: self._get_column_type(col) for col in self.df.columns}
        
        self.date_columns = [col for col, t in self.column_types.items() if t == 'date']
        self.numeric_columns = [col for col, t in self.column_types.items() if t == 'numeric']
        self.categorical_columns = [col for col, t in self.column_types.items() 
                                  if t == 'categorical' and col not in self.date_columns]

    def _create_time_series_plots(self) -> None:
        """Generate time series line charts for all date columns"""
        for date_col in self.date_columns:
            numeric_cols = [col for col in self.numeric_columns if col != date_col]
            if numeric_cols:
                fig = go.Figure()
                for num_col in numeric_cols:
                    fig.add_trace(go.Scatter(
                        x=self.df[date_col],
                        y=self.df[num_col],
                        name=num_col,
                        mode='lines+markers',
                        hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>{num_col}: %{{y}}"
                    ))

                fig.update_layout(
                    title=f"Time Series Analysis: {', '.join(numeric_cols)}",
                    xaxis_title=date_col,
                    yaxis_title="Value",
                    hovermode='x unified',
                    template='plotly_white',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                
                filename = f"{self.output_dir}/timeseries_{date_col}.html"
                self._save_chart(fig, filename)
                self.generated_figures.append(fig)

    def _create_categorical_plots(self) -> None:
        """Generate categorical visualizations with smart aggregation"""
        for cat_col in self.categorical_columns:
            # Generate aggregated bar charts
            agg_df = self.df.groupby(cat_col)[self.numeric_columns].sum().reset_index()
            
            fig = go.Figure()
            for num_col in self.numeric_columns:
                fig.add_trace(go.Bar(
                    x=agg_df[cat_col],
                    y=agg_df[num_col],
                    name=num_col,
                    text=agg_df[num_col],
                    texttemplate='%{text:.2s}',
                    textposition='outside'
                ))

            fig.update_layout(
                title=f"{cat_col} Analysis",
                xaxis_title=cat_col,
                yaxis_title="Total Value",
                barmode='group',
                template='plotly_white',
                uniformtext_minsize=8,
                uniformtext_mode='hide'
            )
            
            filename = f"{self.output_dir}/categorical_{cat_col}.html"
            self._save_chart(fig, filename)
            self.generated_figures.append(fig)

            # Generate pie charts for top categories
            top_categories = self.df[cat_col].value_counts().nlargest(5).index
            filtered_df = self.df[self.df[cat_col].isin(top_categories)]
            
            for num_col in self.numeric_columns:
                pie_fig = go.Figure(go.Pie(
                    labels=filtered_df[cat_col],
                    values=filtered_df[num_col],
                    hole=0.3,
                    textinfo='label+percent',
                    insidetextorientation='radial'
                ))
                
                pie_fig.update_layout(
                    title=f"{num_col} Distribution by {cat_col} (Top 5)",
                    template='plotly_white'
                )
                
                filename = f"{self.output_dir}/pie_{cat_col}_{num_col}.html"
                self._save_chart(pie_fig, filename)
                self.generated_figures.append(pie_fig)

    def _create_numeric_distributions(self) -> None:
        """Create distribution plots for numeric columns"""
        for num_col in self.numeric_columns:
            # Histogram with box plot
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=self.df[num_col],
                name=num_col,
                nbinsx=20,
                marker_color='#1f77b4'
            ))
            
            fig.add_trace(go.Box(
                x=self.df[num_col],
                name=num_col,
                marker_color='#ff7f0e'
            ))

            fig.update_layout(
                title=f"{num_col} Distribution",
                xaxis_title=num_col,
                template='plotly_white',
                showlegend=False
            )
            
            filename = f"{self.output_dir}/distribution_{num_col}.html"
            self._save_chart(fig, filename)
            self.generated_figures.append(fig)

# Example usage
if __name__ == "__main__":
    # Sample dataset
    data = {
        'OrderDate': pd.date_range(start='2023-01-01', periods=100),
        'Sales': np.random.normal(1000, 200, 100).cumsum(),
        'Profit': np.random.normal(200, 50, 100).cumsum(),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'ProductCategory': np.random.choice(['Electronics', 'Furniture', 'Office Supplies'], 100),
        'CustomerRating': np.random.randint(1, 6, 100)
    }
    df = pd.DataFrame(data)
    
    # Generate insights
    plotter = SmartPlotGenerator(df)
    figures = plotter.generate_insights(output_dir="business_insights")
    
    print(f"Generated {len(figures)} insights:")
    print("\n".join(plotter.generated_files))
