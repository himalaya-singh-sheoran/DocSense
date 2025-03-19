import pandas as pd
import numpy as np
import warnings
from scipy import stats
from textwrap import wrap

class DataFrameAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.column_types = self._categorize_columns()
        self.summary = self._generate_summary()
        self.insights = self._generate_insights()

    def _categorize_columns(self):
        """Smart column type classification with enhanced datetime detection"""
        col_types = {'numeric': [], 'categorical': [], 'datetime': []}
        
        for col in self.df.columns:
            # Handle numeric types
            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_types['numeric'].append(col)
                continue
                
            # Attempt datetime conversion
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    converted = pd.to_datetime(
                        self.df[col], 
                        infer_datetime_format=True,
                        errors='coerce'
                    )
                    if converted.notna().mean() > 0.7:
                        self.df[col] = converted
                        col_types['datetime'].append(col)
                        continue
                except: pass
                
            # Treat as categorical
            col_types['categorical'].append(col)
            # Convert to category type if unique ratio < 50%
            if self.df[col].nunique()/len(self.df) < 0.5:
                self.df[col] = self.df[col].astype('category')
                
        return col_types

    def _generate_summary(self):
        """Comprehensive data profile with smart formatting"""
        report = []
        
        # Dataset overview
        report.append(f"📊 Dataset Overview\n{'='*30}")
        report.append(f"• Contains {len(self.df):,} records with {len(self.df.columns)} columns")
        report.append(f"• Time period coverage: {self._get_date_range()}")
        
        # Column type breakdown
        type_counts = {k: len(v) for k,v in self.column_types.items()}
        report.append("\n🔍 Column Types\n" + "\n".join(
            f"• {k.title()}: {v} columns" for k,v in type_counts.items() if v > 0
        ))
        
        # Detailed numeric analysis
        if self.column_types['numeric']:
            report.append("\n🧮 Numerical Summary")
            for col in self.column_types['numeric']:
                stats = self.df[col].describe(percentiles=[.25, .5, .75])
                report.append(
                    f"  → {col}:\n"
                    f"    Mean: {stats['mean']:,.1f} | "
                    f"Range: {stats['min']:,.1f}-{stats['max']:,.1f}\n"
                    f"    Quartiles: {stats['25%']:,.1f} | {stats['50%']:,.1f} | {stats['75%']:,.1f}"
                )
        
        # Enhanced categorical analysis
        if self.column_types['categorical']:
            report.append("\n🏷️ Categorical Distributions")
            for col in self.column_types['categorical']:
                dist = self.df[col].value_counts(normalize=True).head(3)
                report.append(
                    f"  → {col}:\n"
                    f"    Top values: " + " | ".join(
                        f"{k} ({v:.1%})" for k,v in dist.items()
                    ) + f"\n    Unique values: {self.df[col].nunique():,}"
                )
                
        return "\n".join(report)

    def _generate_insights(self):
        """Generate business-focused insights with explanations"""
        insights = []
        
        # Correlation network analysis
        if len(self.column_types['numeric']) > 1:
            insights += self._numeric_relationships()
            
        # Temporal trend analysis
        if self.column_types['datetime']:
            insights += self._temporal_analysis()
            
        # Categorical impact analysis
        if self.column_types['categorical'] and self.column_types['numeric']:
            insights += self._category_impact_analysis()
            
        # Distribution characteristics
        insights += self._distribution_analysis()
            
        # Data quality indicators
        insights += self._data_quality_check()
        
        return insights

    def _numeric_relationships(self):
        """Find meaningful numerical relationships"""
        insights = []
        corr_matrix = self.df[self.column_types['numeric']].corr().abs()
        
        for (col1, col2), value in corr_matrix.stack().items():
            if col1 == col2: continue
            if value > 0.75:
                direction = "positive" if self.df[col1].corr(self.df[col2]) > 0 else "negative"
                insights.append(
                    f"📈 Strong {direction} relationship between {col1} and {col2} "
                    f"(correlation: {value:.2f}). These metrics tend to move together."
                )
                
        return insights

    def _temporal_analysis(self):
        """Identify time-based patterns"""
        insights = []
        for date_col in self.column_types['datetime']:
            if not self.column_types['numeric']: continue
            
            # Analyze most prominent numeric column
            num_col = self.column_types['numeric'][0]
            ts = self.df.set_index(date_col)[num_col].resample('M').mean()
            
            if len(ts) > 2:
                last_3 = ts.pct_change().tail(3).mean()
                overall_trend = ts.pct_change().mean()
                
                insights.append(
                    f"⏳ Temporal Trend: {num_col} shows "
                    f"{'growth' if overall_trend > 0 else 'decline'} trend "
                    f"({abs(overall_trend):.1%} monthly)\n"
                    f"   Recent momentum: Last 3 months average {last_3:.1%} change"
                )
                
        return insights

    def _category_impact_analysis(self):
        """Analyze categorical impact on numeric metrics"""
        insights = []
        for cat_col in self.column_types['categorical']:
            for num_col in self.column_types['numeric']:
                grouped = self.df.groupby(cat_col)[num_col].agg(['mean', 'count'])
                if len(grouped) < 2: continue
                
                # Calculate performance spread
                spread = grouped['mean'].max() - grouped['mean'].min()
                if spread > grouped['mean'].std() * 2:
                    top = grouped['mean'].idxmax()
                    bottom = grouped['mean'].idxmin()
                    insights.append(
                        f"🔍 Category Impact: {cat_col} significantly affects {num_col}\n"
                        f"   • Top performer: {top} ({grouped['mean'][top]:,.1f})\n"
                        f"   • Bottom performer: {bottom} ({grouped['mean'][bottom]:,.1f})\n"
                        f"   • Performance spread: {spread:,.1f} ({spread/grouped['mean'].mean():.1%} of average)"
                    )
                    
        return insights

    def _distribution_analysis(self):
        """Analyze data distribution characteristics"""
        insights = []
        for col in self.column_types['numeric']:
            skew = self.df[col].skew()
            kurtosis = self.df[col].kurtosis()
            
            analysis = []
            if abs(skew) > 1:
                analysis.append(f"skewed {'right' if skew > 0 else 'left'}")
            if kurtosis > 3:
                analysis.append("heavy-tailed")
            elif kurtosis < 1:
                analysis.append("light-tailed")
                
            if analysis:
                insights.append(
                    f"📊 Distribution Alert: {col} shows {', '.join(analysis)} distribution\n"
                    f"   (Skewness: {skew:.2f}, Kurtosis: {kurtosis:.2f})"
                )
                
        return insights

    def _data_quality_check(self):
        """Identify potential data quality issues"""
        insights = []
        # High cardinality check
        for col in self.column_types['categorical']:
            if self.df[col].nunique()/len(self.df) > 0.9:
                insights.append(
                    f"🛑 Data Quality: {col} has high uniqueness "
                    f"({self.df[col].nunique():,} unique values)\n"
                    "   May contain identifiers or free-form text"
                )
                
        # Missing values check
        missing = self.df.isna().mean()
        for col, pct in missing.items():
            if pct > 0.2:
                insights.append(
                    f"⚠️ Missing Values: {col} has {pct:.1%} missing data\n"
                    "   Consider imputation or investigation"
                )
                
        return insights

    def _get_date_range(self):
        """Format date range information"""
        if not self.column_types['datetime']: return "No date columns identified"
        
        ranges = []
        for col in self.column_types['datetime']:
            dates = self.df[col].dropna()
            if not dates.empty:
                ranges.append(f"{col}: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
                
        return "\n  ".join(ranges) if ranges else "No valid date ranges detected"

    def generate_report(self, line_width=80):
        """Professional formatted report with text wrapping"""
        report = [
            "COMPREHENSIVE DATA ANALYSIS REPORT",
            "="*line_width,
            self.summary,
            "\nKEY INSIGHTS & RECOMMENDATIONS:",
            "="*line_width
        ]
        
        # Wrap long text lines
        wrapped_insights = []
        for insight in self.insights:
            wrapped = wrap(insight, width=line_width-3)
            wrapped_insights.append("\n  ".join(wrapped))
            
        return "\n".join(report + wrapped_insights)

# Example Usage
if __name__ == "__main__":
    # Create test data
    data = {
        'sales': np.random.normal(50000, 15000, 100),
        'profit': np.random.lognormal(8, 0.3, 100),
        'region': np.random.choice(['North', 'South', 'West', 'East'], 100),
        'order_date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'customer_type': np.random.choice(['Retail', 'Wholesale', 'Online'], 100)
    }
    
    df = pd.DataFrame(data)
    analyzer = DataFrameAnalyzer(df)
    print(analyzer.generate_report())
