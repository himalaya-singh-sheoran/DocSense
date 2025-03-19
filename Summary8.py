import pandas as pd
import numpy as np
from scipy.stats import linregress, zscore
from textwrap import wrap
import warnings

class DataFrameAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.column_types = self._categorize_columns()
        self.insights = []
        self._generate_all_insights()
        
    def _categorize_columns(self):
        """Classify columns into numeric, categorical, and datetime"""
        col_types = {'numeric': [], 'categorical': [], 'datetime': []}
        
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_types['numeric'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                col_types['datetime'].append(col)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        converted = pd.to_datetime(self.df[col], errors='coerce')
                        if converted.notna().mean() > 0.7:
                            self.df[col] = converted
                            col_types['datetime'].append(col)
                        else:
                            col_types['categorical'].append(col)
                    except:
                        col_types['categorical'].append(col)
        return col_types

    def _generate_all_insights(self):
        """Orchestrate insight generation in logical order"""
        self._basic_statistics()
        self._temporal_analysis()
        self._value_concentration()
        self._cross_column_context()
        self._seasonality_analysis()
        self._performance_consistency()
        self._comparative_analysis()
        self._growth_acceleration()
        self._distribution_analysis()
        self._data_quality_check()

    def _basic_statistics(self):
        """Fundamental metrics overview"""
        # Numeric ranges
        for col in self.column_types['numeric']:
            stats = self.df[col].describe()
            self.insights.append(
                f"📊 {col} ranges from {stats['min']:,.0f} to {stats['max']:,.0f} "
                f"(Avg: {stats['mean']:,.0f}, Median: {stats['50%']:,.0f})"
            )
        
        # Category distributions
        for col in self.column_types['categorical']:
            counts = self.df[col].value_counts(normalize=True)
            if not counts.empty:
                top = counts.index[0]
                freq = counts.iloc[0]
                self.insights.append(
                    f"🏷️ Most common {col}: '{top}' ({freq:.1%}) "
                    f"among {len(counts):,} unique values"
                )

    def _temporal_analysis(self):
        """Time-based patterns and trends"""
        if not self.column_types['datetime']: return
        
        for date_col in self.column_types['datetime']:
            dates = self.df[date_col].dropna()
            if dates.empty: continue
            
            date_range = f"{dates.min().strftime('%d %b %Y')} - {dates.max().strftime('%d %b %Y')}"
            self.insights.append(f"⏳ Date range for {date_col}: {date_range}")
            
            for num_col in self.column_types['numeric']:
                ts = self.df.set_index(date_col)[num_col].resample('D').mean().ffill()
                if len(ts) < 2: continue
                
                # Trend analysis
                slope, _, _, _, _ = linregress(np.arange(len(ts)), ts)
                trend_dir = "upward" if slope > 0 else "downward"
                total_change = ((ts.iloc[-1] - ts.iloc[0])/ts.iloc[0]
                
                self.insights.append(
                    f"📈 {num_col} shows {trend_dir} trend ({total_change:+.1%}) "
                    f"over {len(ts):,} days"
                )

    def _value_concentration(self):
        """Top contributor analysis"""
        if not (self.column_types['categorical'] and self.column_types['numeric']): return
        
        for cat_col in self.column_types['categorical']:
            for num_col in self.column_types['numeric']:
                grouped = self.df.groupby(cat_col)[num_col].sum().nlargest(3)
                if len(grouped) < 1: continue
                
                total_contribution = grouped.sum()/self.df[num_col].sum()
                self.insights.append(
                    f"💰 Top 3 {cat_col} account for {total_contribution:.1%} "
                    f"of total {num_col}: {', '.join(grouped.index.astype(str))}"
                )

    def _cross_column_context(self):
        """Relationships between columns"""
        for num_col in self.column_types['numeric']:
            max_val = self.df[num_col].max()
            max_row = self.df.loc[self.df[num_col].idxmax()]
            
            context = []
            for cat_col in self.column_types['categorical']:
                context.append(f"{cat_col}='{max_row[cat_col]}'")
            for dt_col in self.column_types['datetime']:
                context.append(f"on {max_row[dt_col].strftime('%d %b %Y')}")
                
            self.insights.append(
                f"🏆 Peak {num_col} ({max_val:,.0f}) occurred with: " + 
                ", ".join(context)
            )

    def _seasonality_analysis(self):
        """Weekly/monthly patterns"""
        if not self.column_types['datetime']: return
        
        for date_col in self.column_types['datetime']:
            self.df['weekday'] = self.df[date_col].dt.day_name()
            self.df['month'] = self.df[date_col].dt.month_name()
            
            for num_col in self.column_types['numeric']:
                # Weekly pattern
                weekly = self.df.groupby('weekday')[num_col].mean()
                if weekly.max()/weekly.min() > 1.5:
                    peak_day = weekly.idxmax()
                    self.insights.append(
                        f"📅 {num_col} averages {weekly[peak_day]/weekly.mean():.0%} higher "
                        f"on {peak_day}s compared to weekly average"
                    )
                
                # Monthly pattern
                monthly = self.df.groupby('month')[num_col].mean()
                if monthly.max()/monthly.min() > 2:
                    peak_month = monthly.idxmax()
                    self.insights.append(
                        f"🌞 Seasonal peak: {num_col} in {peak_month} is "
                        f"{monthly[peak_month]/monthly.mean():.0%} above annual average"
                    )

    def _performance_consistency(self):
        """Consistent top performers"""
        if not (self.column_types['categorical'] and len(self.column_types['numeric']) >1): return
        
        for cat_col in self.column_types['categorical']:
            top_performers = []
            for num_col in self.column_types['numeric']:
                tops = self.df.groupby(cat_col)[num_col].mean().nlargest(3).index
                top_performers.extend(tops)
            
            counter = pd.Series(top_performers).value_counts()
            consistent = counter[counter == len(self.column_types['numeric'])].index
            if not consistent.empty:
                self.insights.append(
                    f"🎯 Consistent performer: {cat_col} '{consistent[0]}' "
                    f"ranks top 3 in all {len(self.column_types['numeric'])} metrics"
                )

    def _comparative_analysis(self):
        """Recent vs historical comparison"""
        if not self.column_types['datetime']: return
        
        date_col = self.column_types['datetime'][0]
        cutoff = self.df[date_col].max() - pd.DateOffset(days=30)
        
        for num_col in self.column_types['numeric']:
            recent = self.df[self.df[date_col] > cutoff][num_col].mean()
            historical = self.df[self.df[date_col] <= cutoff][num_col].mean()
            
            if pd.notna(recent) and pd.notna(historical) and historical !=0:
                change = (recent - historical)/historical
                self.insights.append(
                    f"🔔 Recent {num_col} {'↑' if change>0 else '↓'} "
                    f"{abs(change):.0%} vs previous period "
                    f"({historical:,.0f} → {recent:,.0f})"
                )

    def _growth_acceleration(self):
        """Growth rate changes"""
        if not self.column_types['datetime']: return
        
        for date_col in self.column_types['datetime']:
            for num_col in self.column_types['numeric']:
                ts = self.df.set_index(date_col)[num_col].resample('M').mean()
                if len(ts) < 4: continue
                
                growth = ts.pct_change()
                if growth.mean() > 0.1 and growth.iloc[-1] > growth.mean():
                    self.insights.append(
                        f"🚀 Accelerating growth: {num_col} latest month "
                        f"↑{growth.iloc[-1]:.0%} vs average ↑{growth.mean():.0%}"
                    )

    def _distribution_analysis(self):
        """Data distribution characteristics"""
        for num_col in self.column_types['numeric']:
            skewness = self.df[num_col].skew()
            if abs(skewness) > 1:
                self.insights.append(
                    f"📉 {num_col} distribution is skewed "
                    f"{'right' if skewness>0 else 'left'} (most values "
                    f"below {'high' if skewness>0 else 'low'} end)"
                )

    def _data_quality_check(self):
        """Data health indicators"""
        # Missing values
        missing = self.df.isna().mean()
        for col, pct in missing.items():
            if pct > 0.2:
                self.insights.append(
                    f"⚠️ Caution: {col} has {pct:.0%} missing values"
                )
        
        # High cardinality
        for col in self.column_types['categorical']:
            if self.df[col].nunique()/len(self.df) > 0.8:
                self.insights.append(
                    f"🛑 Check: {col} has {self.df[col].nunique():,} unique values "
                    "- possible free-form text"
                )

    def generate_report(self, line_width=80):
        """Formatted business report"""
        sections = [
            "COMPREHENSIVE DATA ANALYSIS REPORT",
            "="*line_width,
            "🔍 Key Insights:",
            *self.insights,
            "\nℹ️ Data Quality Notes:",
            "="*line_width,
            f"• Total records: {len(self.df):,}",
            f"• Columns analyzed: {len(self.column_types['numeric'])} numeric, "
            f"{len(self.column_types['categorical'])} categorical, "
            f"{len(self.column_types['datetime'])} datetime"
        ]
        
        wrapped = []
        for line in sections:
            if isinstance(line, str):
                wrapped.extend(wrap(line, width=line_width))
            else:
                wrapped.append(line)
        
        return "\n".join(wrapped)

# Example Usage
if __name__ == "__main__":
    data = {
        'sales': [120, 150, 95, 200, 180, 210, 300, 250, 190, 220],
        'profit': [15, 18, 9, 25, 22, 28, 35, 30, 24, 27],
        'region': ['North', 'South', 'West', 'North', 'South', 'East', 'East', 'West', 'North', 'South'],
        'category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A'],
        'order_date': pd.date_range('2023-01-01', periods=10, freq='W')
    }
    
    df = pd.DataFrame(data)
    analyzer = DataFrameAnalyzer(df)
    print(analyzer.generate_report())
