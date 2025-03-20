import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
from typing import List

class AutoPlotGenerator:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.default_template = 'plotly_white+gridon'
        self.generated_files = []
        
    def _save_chart(self, fig: go.Figure, filename: str) -> None:
        """Save charts and track generated files"""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))
        self.generated_files.append(str(path.absolute()))
        
    def _get_datetime_cols(self) -> List[str]:
        """Identify datetime columns"""
        return self.df.select_dtypes(include=['datetime64']).columns.tolist()
    
    def _get_numeric_cols(self) -> List[str]:
        """Identify numeric columns"""
        return self.df.select_dtypes(include=['number']).columns.tolist()
    
    def _get_categorical_cols(self, max_unique: int = 15) -> List[str]:
        """Identify categorical columns (non-numeric with limited unique values)"""
        datetime_cols = self._get_datetime_cols()
        return [col for col in self.df.columns 
                if col not in datetime_cols and 
                (self.df[col].dtype == 'object' or 
                 self.df[col].nunique() <= max_unique)]
    
    def _generate_combo_name(self, cols: List[str]) -> str:
        """Generate filename-safe combination name"""
        return "_".join(cols).replace(" ", "_").lower()
    
    def auto_generate_plots(self) -> List[go.Figure]:
        """Automatically generate and save appropriate plots"""
        figures = []
        
        # Get column classifications
        datetime_cols = self._get_datetime_cols()
        numeric_cols = self._get_numeric_cols()
        categorical_cols = self._get_categorical_cols()
        
        # 1. Time Series Charts
        for dt_col in datetime_cols:
            if numeric_cols:
                # Line chart for time series
                line_fig = self._create_line_chart(
                    x_col=dt_col,
                    y_cols=numeric_cols,
                    title=f"Time Series: {', '.join(numeric_cols)}"
                )
                figures.append(line_fig)
                
                # Bar chart for time comparisons
                bar_fig = self._create_bar_chart(
                    x_col=dt_col,
                    y_cols=numeric_cols,
                    title=f"Time Distribution: {', '.join(numeric_cols)}"
                )
                figures.append(bar_fig)
        
        # 2. Categorical-Numeric Relationships
        for cat_col in categorical_cols:
            if numeric_cols:
                # Grouped bar charts
                bar_fig = self._create_bar_chart(
                    x_col=cat_col,
                    y_cols=numeric_cols,
                    title=f"{cat_col} Distribution"
                )
                figures.append(bar_fig)
            
            # Pie charts for single numeric comparison
            for num_col in numeric_cols:
                pie_fig = self._create_pie_chart(
                    labels_col=cat_col,
                    values_col=num_col,
                    title=f"{cat_col} Composition by {num_col}"
                )
                figures.append(pie_fig)
        
        # 3. Numeric-Numeric Relationships
        if len(numeric_cols) >= 2:
            # Scatter matrix (extension beyond basic types)
            pass  # Could add additional visualization types here
        
        return figures
    
    def _create_bar_chart(self, x_col: str, y_cols: List[str], title: str) -> go.Figure:
        """Helper to create bar chart"""
        fig = go.Figure()
        for y_col in y_cols:
            fig.add_trace(go.Bar(
                x=self.df[x_col],
                y=self.df[y_col],
                name=y_col,
                textposition='auto'
            ))
        
        fig.update_layout(
            title=title,
            template=self.default_template,
            barmode='group' if len(y_cols) > 1 else 'stack',
            xaxis_title=x_col,
            yaxis_title="Value"
        )
        filename = f"bar_{self._generate_combo_name([x_col]+y_cols)}.html"
        self._save_chart(fig, filename)
        return fig
    
    def _create_line_chart(self, x_col: str, y_cols: List[str], title: str) -> go.Figure:
        """Helper to create line chart"""
        fig = go.Figure()
        for y_col in y_cols:
            fig.add_trace(go.Scatter(
                x=self.df[x_col],
                y=self.df[y_col],
                mode='lines+markers',
                name=y_col,
                hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}"
            ))
        
        fig.update_layout(
            title=title,
            template=self.default_template,
            xaxis_title=x_col,
            yaxis_title="Value",
            hovermode='x unified'
        )
        filename = f"line_{self._generate_combo_name([x_col]+y_cols)}.html"
        self._save_chart(fig, filename)
        return fig
    
    def _create_pie_chart(self, labels_col: str, values_col: str, title: str) -> go.Figure:
        """Helper to create pie chart"""
        fig = go.Figure(go.Pie(
            labels=self.df[labels_col],
            values=self.df[values_col],
            textinfo='label+percent',
            hoverinfo='value+percent',
            hole=0.3
        ))
        
        fig.update_layout(
            title=title,
            template=self.default_template
        )
        filename = f"pie_{self._generate_combo_name([labels_col, values_col])}.html"
        self._save_chart(fig, filename)
        return fig

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        'Date': pd.date_range(start='2023-01-01', periods=10),
        'Sales': [100, 120, 130, 115, 140, 160, 170, 165, 180, 200],
        'Profit': [30, 35, 40, 32, 42, 50, 55, 52, 60, 65],
        'Region': ['North', 'South'] * 5,
        'Product': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    df = pd.DataFrame(data)
    
    # Generate and save plots
    plotter = AutoPlotGenerator(df)
    generated_plots = plotter.auto_generate_plots()
    
    print(f"Generated {len(generated_plots)} plots:")
    print("\n".join(plotter.generated_files))
  
