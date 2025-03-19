class DataFrameAnalyzer:
    # ... (Keep previous methods unchanged) ...

    def _generate_insights(self):
        """Generate comprehensive business insights"""
        insights = []
        
        # Existing analyses
        insights += self._basic_statistics()
        insights += self._temporal_analysis()
        insights += self._cross_column_analysis()
        insights += self._value_concentration()
        insights += self._seasonality_analysis()
        insights += self._performance_consistency()
        insights += self._comparative_analysis()
        
        return insights

    def _value_concentration(self):
        """Analyze value distribution across categories"""
        insights = []
        if self.column_types['categorical'] and self.column_types['numeric']:
            for cat_col in self.column_types['categorical']:
                for num_col in self.column_types['numeric']:
                    # Calculate top 3 category contribution
                    total = self.df[num_col].sum()
                    top_categories = self.df.groupby(cat_col)[num_col].sum().nlargest(3)
                    contribution = top_categories.sum() / total
                    
                    if contribution > 0.7:
                        insights.append(
                            f"💰 Top 3 {cat_col} categories account for {contribution:.0%} "
                            f"of total {num_col} ({', '.join(top_categories.index)})"
                        )

        return insights

    def _seasonality_analysis(self):
        """Identify weekly/monthly patterns"""
        insights = []
        if self.column_types['datetime'] and self.column_types['numeric']:
            for date_col in self.column_types['datetime']:
                for num_col in self.column_types['numeric']:
                    # Daily patterns
                    self.df['day_of_week'] = self.df[date_col].dt.day_name()
                    weekday_avg = self.df.groupby('day_of_week')[num_col].mean()
                    
                    if weekday_avg.max() / weekday_avg.min() > 1.5:
                        best_day = weekday_avg.idxmax()
                        insights.append(
                            f"📅 {num_col} tends to be {weekday_avg[best_day]/weekday_avg.mean():.0%} "
                            f"higher than average on {best_day}s"
                        )
                    
                    # Monthly patterns
                    self.df['month'] = self.df[date_col].dt.month_name()
                    monthly_avg = self.df.groupby('month')[num_col].mean()
                    
                    if monthly_avg.max() / monthly_avg.min() > 2:
                        best_month = monthly_avg.idxmax()
                        insights.append(
                            f"🌞 {num_col} peaks in {best_month} - "
                            f"{monthly_avg[best_month]/monthly_avg.mean():.0%} above yearly average"
                        )
        
        return insights

    def _performance_consistency(self):
        """Identify consistently performing categories"""
        insights = []
        if self.column_types['categorical'] and len(self.column_types['numeric']) > 1:
            performance_matrix = {}
            
            for cat_col in self.column_types['categorical']:
                matrix = []
                for num_col in self.column_types['numeric']:
                    top_categories = self.df.groupby(cat_col)[num_col].mean().nlargest(3).index
                    matrix.append(set(top_categories))
                
                # Find common top performers
                common = set.intersection(*matrix)
                if common:
                    insights.append(
                        f"🏆 {cat_col} categories {', '.join(common)} "
                        "consistently rank in top 3 across all metrics"
                    )
        
        return insights

    def _comparative_analysis(self):
        """Compare current vs historical performance"""
        insights = []
        if self.column_types['datetime'] and self.column_types['numeric']:
            for date_col in self.column_types['datetime']:
                for num_col in self.column_types['numeric']:
                    # Compare last 30 days vs previous period
                    recent = self.df[self.df[date_col] > (self.df[date_col].max() - pd.DateOffset(days=30))
                    prev = self.df[self.df[date_col] < (self.df[date_col].max() - pd.DateOffset(days=30))]
                    
                    if len(recent) > 0 and len(prev) > 0:
                        change = (recent[num_col].mean() - prev[num_col].mean()) / prev[num_col].mean()
                        insights.append(
                            f"🔔 Recent {num_col} is {change:+.0%} compared to previous period "
                            f"({prev[num_col].mean():.0f} → {recent[num_col].mean():.0f})"
                        )
        
        return insights

    def _growth_analysis(self):
        """Analyze growth rates and acceleration"""
        insights = []
        if self.column_types['datetime'] and self.column_types['numeric']:
            for date_col in self.column_types['datetime']:
                for num_col in self.column_types['numeric']:
                    ts = self.df.set_index(date_col)[num_col].resample('M').mean()
                    if len(ts) > 3:
                        growth_rates = ts.pct_change()
                        if growth_rates.mean() > 0.1 and growth_rates.iloc[-1] > growth_rates.mean():
                            insights.append(
                                f"🚀 {num_col} growth is accelerating - "
                                f"last month increased by {growth_rates.iloc[-1]:.0%} "
                                f"(average {growth_rates.mean():.0%})"
                            )
        
        return insights
