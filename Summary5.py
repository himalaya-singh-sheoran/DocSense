class DataFrameInsightGenerator:
    # ... [Keep previous methods unchanged until _generate_insights] ...

    def _generate_insights(self):
        """Generate actionable business insights with enhanced categorical analysis"""
        insights = []
        
        # Existing numeric and datetime insights
        # ... [Keep previous numeric/datetime analysis here] ...
        
        # Enhanced categorical analysis
        if self.column_types['categorical']:
            insights.append("\n🔠 CATEGORY INSIGHTS:")
            
            for col in self.column_types['categorical']:
                # 1. Frequency distribution analysis
                value_counts = self.df[col].value_counts(normalize=True)
                top_3 = value_counts.head(3)
                insights.append(
                    f"• {col} distribution: " + 
                    " | ".join([f"{cat} ({pct:.1%})" for cat, pct in top_3.items()])
                )
                
                # 2. Rare category detection (less than 5% occurrence)
                rare_categories = value_counts[value_counts < 0.05]
                if not rare_categories.empty:
                    insights.append(
                        f"   - Rare entries: {len(rare_categories)} categories " +
                        f"with <5% occurrence ({', '.join(rare_categories.index.astype(str).tolist()[:3])}...)"
                    )
                
                # 3. Cross-category relationships
                for other_col in self.column_types['categorical']:
                    if col != other_col:
                        crosstab = pd.crosstab(self.df[col], self.df[other_col], normalize='all') * 100
                        max_combination = crosstab.stack().idxmax()
                        max_value = crosstab.stack().max()
                        if max_value > 20:  # Only show significant combinations
                            insights.append(
                                f"   - Strong association: " +
                                f"'{max_combination[0]}' in {col} " +
                                f"often occurs with '{max_combination[1]}' in {other_col} " +
                                f"({max_value:.1f}% of cases)"
                            )
                
                # 4. Temporal patterns (if datetime exists)
                if self.column_types['datetime']:
                    time_col = self.column_types['datetime'][0]
                    time_trend = self.df.groupby(
                        [pd.Grouper(key=time_col, freq='M'), col]
                    ).size().unstack().fillna(0)
                    
                    # Find categories with significant trends
                    for cat in time_trend.columns:
                        trend = time_trend[cat].pct_change().mean()
                        if abs(trend) > 0.1:  # 10% monthly growth/decline
                            insights.append(
                                f"   - '{cat}' shows " +
                                f"{'growth' if trend > 0 else 'decline'} " +
                                f"({abs(trend):.1%} monthly change)"
                            )
                
                # 5. Cardinality check
                unique_count = self.df[col].nunique()
                if unique_count > len(self.df) * 0.5:
                    insights.append(
                        f"   - High uniqueness: " +
                        f"{unique_count} distinct values " +
                        f"(potentially identifiers or free-form text)"
                    )

        return insights

    def show_report(self):
        """Generate final formatted report"""
        report = [
            "📊 DATA OVERVIEW",
            self.summary,
            "\n🔍 KEY INSIGHTS",
            *self.insights
        ]
        return "\n".join(report)
