# Complete analysis.py that matches your notebook workflow
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


class SMEeBayAnalyzer:
    def __init__(self, data_path: str):
        """
        Initialize with CSV file path. Handles German→English translation and data cleaning
        as demonstrated in your notebook.
        """
        self.data_path = data_path
        self.results = {}
        self.load_and_prepare_data()

    def load_and_prepare_data(self):
        """Load and clean the eBay campaign data with German→English translation"""
        try:
            # Try simple loading first
            df = pd.read_csv(self.data_path)
            print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
            
        except Exception as e:
            # If simple loading fails, use complex parsing from your notebook
            print(f"Simple loading failed: {e}")
            print("Attempting complex parsing...")
            
            with open(self.data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Extract header from line 3 (as in your notebook)
            header_line = lines[2].strip().replace('"', '')
            headers = [col.strip() for col in header_line.split(';')]
            
            # Parse data rows
            data_rows = []
            for i in range(3, len(lines)):
                line = lines[i].strip()
                if line and line != '""':
                    row_data = [cell.strip().replace('"', '') for cell in line.split(';')]
                    if len(row_data) == len(headers):
                        data_rows.append(row_data)
            
            df = pd.DataFrame(data_rows, columns=headers)
            
            # German → English translation (from your notebook)
            german_to_english = {
                'Startdatum': 'Start_Date',
                'Enddatum': 'End_Date', 
                'Name der Kampagne': 'Campaign_Name',
                'Kampagnen-ID': 'Campaign_ID',
                'Startdatum der Kampagne': 'Campaign_Start_Date',
                'Enddatum der Kampagne': 'Campaign_End_Date',
                'Status': 'Status',
                'Artikelnr.': 'Product_ID',
                'Titel': 'Product_Title',
                'Angebotsformat': 'Listing_Format',
                'Preis (aktueller oder zuletzt angegebener)': 'Current_Price',
                'Verfügbare Stückzahl': 'Available_Stock',
                'Angebotsstart': 'Listing_Start',
                'Angebotsende': 'Listing_End',
                'Anzeigen-Impressions (über Platzierungen bei eBay)': 'Ad_Impressions_eBay',
                'Anzeigen-Klicks insgesamt': 'Total_Ad_Clicks',
                'Verkauft mit Anzeigen - Gesamtstückzahl': 'Units_Sold_with_Ads_Total',
                'Organisch verkauft - Stückzahl': 'Units_Sold_Organic',
                'Verkauft - Gesamtstückzahl': 'Total_Units_Sold',
                'Anzeigen-Konversionsrate (Verkauft mit Anzeigen - Stückzahl/Anzeigen-Klicks)': 'Ad_Conversion_Rate',
                'Anzeigen-Verteilung (Verkauft mit Anzeigen - Stückzahl/Verkauft - Gesamtstückzahl)': 'Ad_Attribution_Rate',
                'Gesamtumsatz mit Anzeigen': 'Total_Revenue_with_Ads',
                'Anzeigengebühren (ohne MwSt.)': 'Ad_Spend_excl_VAT',
                'Rentabilität der Anzeigenkosten (Umsatz/Anzeigengebühren (ohne MwSt.))': 'ROAS',
                'Durchschn. Kosten pro Verkauf': 'Avg_Cost_Per_Sale',
                'Anzeigen-Klicks (über Platzierungen bei eBay)': 'Ad_Clicks_eBay',
                'Anzeigen-Klicks (über externe Platzierungen)': 'Ad_Clicks_External',
                'Mit Anzeigen verkaufte Stückzahl (über Platzierungen bei eBay)': 'Units_Sold_Ads_eBay',
                'Mit Anzeigen verkaufte Stückzahl (über externe Platzierungen)': 'Units_Sold_Ads_External',
                'Anzeigen-Umsatz (über Platzierungen bei eBay)': 'Ad_Revenue_eBay',
                'Anzeigen-Umsatz (über externe Platzierungen)': 'Ad_Revenue_External',
                'Anzeigengebühren ohne MwSt. (über Platzierungen bei eBay)': 'Ad_Spend_eBay_excl_VAT',
                'Anzeigengebühren ohne MwSt. (über externe Platzierungen)': 'Ad_Spend_External_excl_VAT',
                'Anzeigen-Konversionsrate (über Platzierungen bei eBay)': 'Ad_Conversion_Rate_eBay',
                'Anzeigen-Konversionsrate (über externe Platzierungen)': 'Ad_Conversion_Rate_External',
                'Organische Klicks': 'Organic_Clicks',
                'Organische Impressions': 'Organic_Impressions'
            }
            
            df.columns = [german_to_english.get(col, col) for col in df.columns]
        
        # Numeric cleaning (as in your notebook)
        def clean_numeric(value):
            if pd.isna(value) or value in ["", "--"]:
                return 0
            value = str(value).replace("€","").replace("%","").replace(",",".").strip()
            try:
                return float(value)
            except:
                return 0
        
        # List of numeric columns from your notebook
        num_cols = [
            'Ad_Impressions_eBay','Total_Ad_Clicks','Units_Sold_with_Ads_Total',
            'Units_Sold_Organic','Total_Units_Sold','Ad_Conversion_Rate','Ad_Attribution_Rate',
            'Total_Revenue_with_Ads','Ad_Spend_excl_VAT','ROAS','Avg_Cost_Per_Sale',
            'Ad_Clicks_eBay','Ad_Clicks_External','Units_Sold_Ads_eBay','Units_Sold_Ads_External',
            'Ad_Revenue_eBay','Ad_Revenue_External','Ad_Spend_eBay_excl_VAT','Ad_Spend_External_excl_VAT',
            'Ad_Conversion_Rate_eBay','Ad_Conversion_Rate_External','Organic_Clicks',
            'Organic_Impressions','Current_Price','Available_Stock'
        ]
        
        # Clean numeric columns
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric)
        
        # Calculate derived metrics (as in your notebook)
        df['CTR'] = np.where(df['Ad_Impressions_eBay']>0,
                            df['Total_Ad_Clicks']/df['Ad_Impressions_eBay']*100, 0)
        
        df['CPC'] = np.where(df['Total_Ad_Clicks']>0,
                            df['Ad_Spend_excl_VAT']/df['Total_Ad_Clicks'], 0)
        
        df['Revenue_Per_Click'] = np.where(df['Total_Ad_Clicks']>0,
                                          df['Total_Revenue_with_Ads']/df['Total_Ad_Clicks'], 0)
        
        df['Ad_Efficiency'] = np.where(df['Ad_Spend_excl_VAT']>0,
                                      df['Total_Revenue_with_Ads']/df['Ad_Spend_excl_VAT'], 0)
        
        df['Has_Revenue'] = (df['Total_Revenue_with_Ads']>0).astype(int)
        df['Has_Sales'] = (df['Total_Units_Sold']>0).astype(int)
        df['Organic_Ratio'] = np.where(df['Total_Units_Sold']>0,
                                      df['Units_Sold_Organic']/df['Total_Units_Sold']*100, 0)
        
        self.data = df
        self.prepare_analysis_data()
        print(f"Data cleaning completed. Final shape: {self.data.shape}")

    def prepare_analysis_data(self):
        """Filter and prepare data for analysis"""
        # Filter campaigns with spend and clicks (as in your notebook)
        d = self.data[
            (self.data['Ad_Spend_excl_VAT'] > 0) & 
            (self.data['Total_Ad_Clicks'] > 0)
        ].copy()
        
        # Log transforms
        d['log_ad_spend'] = np.log(d['Ad_Spend_excl_VAT'])
        d['log_revenue'] = np.log(d['Total_Revenue_with_Ads'] + 1)
        
        self.d = d
        print(f"Analysis data prepared: {len(d)} campaigns with ad spend")

    def calculate_vif(self):
        """Calculate Variance Inflation Factors for multicollinearity diagnosis"""
        X = sm.add_constant(self.d[['Ad_Spend_excl_VAT', 'Total_Ad_Clicks', 'CTR', 'CPC']])
        vif = pd.DataFrame()
        vif["Variable"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print("\n🔍 VARIANCE INFLATION FACTORS:")
        print(vif)
        return vif

    def linear_regression(self):
        """Multiple Linear Regression Analysis"""
        y = self.d['Total_Revenue_with_Ads']
        X = sm.add_constant(self.d[['Ad_Spend_excl_VAT', 'Total_Ad_Clicks', 'CTR', 'CPC']])
        m = sm.OLS(y, X).fit()
        print("\n📊 MODEL 1: MULTIPLE LINEAR REGRESSION")
        print(m.summary())
        self.results['linear'] = m
        return m

    def elasticity(self):
        """Log-Log Elasticity Analysis"""
        y = self.d['log_revenue']
        X = sm.add_constant(self.d[['log_ad_spend']])
        m = sm.OLS(y, X).fit()
        e = m.params['log_ad_spend']
        print("\n📈 MODEL 2: ELASTICITY ANALYSIS")
        print(m.summary())
        print(f"Elasticity = {e:.3f} → {'Inelastic' if e < 1 else 'Elastic'} demand")
        self.results['elasticity'] = m
        return m

    def logistic_regression(self):
        """Logistic Regression for Success Probability"""
        y = self.data['Has_Revenue']
        X = sm.add_constant(self.data[['Ad_Spend_excl_VAT', 'CTR', 'Total_Ad_Clicks', 'CPC']])
        m = sm.Logit(y, X).fit(disp=0)
        auc = roc_auc_score(y, m.predict(X))
        print("\n🎯 MODEL 3: SUCCESS PROBABILITY (LOGIT)")
        print(m.summary())
        print(f"AUC = {auc:.3f}")
        self.results['logit'] = m
        return m

    def diminishing_returns(self):
        """Polynomial Regression for Diminishing Returns"""
        X = self.d[['Ad_Spend_excl_VAT']].values
        y = self.d['Total_Revenue_with_Ads'].values
        poly = PolynomialFeatures(2).fit_transform(X)
        m = LinearRegression().fit(poly, y)
        r2 = r2_score(y, m.predict(poly))
        rmse = np.sqrt(mean_squared_error(y, m.predict(poly)))
        quad = m.coef_[2]
        opt = -m.coef_[1] / (2 * quad) if quad < 0 else None
        print("\n📉 MODEL 4: DIMINISHING RETURNS")
        print(f"R²={r2:.3f}, RMSE={rmse:.2f}, Intercept={m.intercept_:.2f}, "
              f"Linear={m.coef_[1]:.2f}, Quad={quad:.6f}")
        if opt:
            print(f"Optimal spend ≈ {opt:.2f}€")
        self.results['poly'] = m
        return m

    def insights(self):
        """Business Insights Summary"""
        print("\n💼 BUSINESS INSIGHTS")
        avg_roi = self.data[self.data['ROAS'] > 0]['ROAS'].mean()
        success_rate = (self.data['Has_Revenue'] == 1).mean() * 100
        avg_cpc = self.data[self.data['CPC'] > 0]['CPC'].mean()
        print(f"ROI: {avg_roi:.2f}€/€ | Success Rate: {success_rate:.1f}% | Avg CPC: {avg_cpc:.3f}€")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("🔍 SME DIGITAL MARKETING ANALYSIS")
        print("=" * 50)
        
        # VIF Check
        self.calculate_vif()
        
        # All Models
        self.linear_regression()
        self.elasticity()
        self.logistic_regression() 
        self.diminishing_returns()
        self.insights()
        
        print("\n✅ Analysis completed successfully!")


if __name__ == "__main__":
    # Example usage - users would run this exactly
    try:
        analyzer = SMEeBayAnalyzer("sample_campaigns.csv")
        analyzer.run_full_analysis()
    except FileNotFoundError:
        print("❌ Error: data/sample_campaigns.csv not found")
        print("Make sure you have the cleaned dataset in the data/ folder")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check that your CSV file matches the expected format")
