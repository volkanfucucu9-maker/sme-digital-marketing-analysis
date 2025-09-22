import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Variance Inflation Factor (VIF) – check multicollinearity
X = sm.add_constant(df_clean[['Ad_Spend_excl_VAT','Total_Ad_Clicks','CTR','CPC']])

vif = pd.DataFrame()

vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)
# Make sure df_clean is defined before running this.
X = sm.add_constant(df_clean[['Ad_Spend_excl_VAT', 'Total_Ad_Clicks', 'CTR', 'CPC']])
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factors:\n", vif)

class SMEeBayAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.results = {}
        self.prepare_data()

    def prepare_data(self):
        # Filter campaigns with spend and clicks
        d = self.data[
            (self.data['Ad_Spend_excl_VAT'] > 0) &
            (self.data['Total_Ad_Clicks'] > 0)
        ].copy()

        # Log transforms
        d['log_ad_spend'] = np.log(d['Ad_Spend_excl_VAT'])
        d['log_revenue'] = np.log(d['Total_Revenue_with_Ads'] + 1)

        self.d = d
        print(f"Analysis data prepared: {len(d)} campaigns with ad spend")

    def linear_regression(self):
        y = self.d['Total_Revenue_with_Ads']
        X = sm.add_constant(self.d[['Ad_Spend_excl_VAT', 'Total_Ad_Clicks', 'CTR', 'CPC']])
        m = sm.OLS(y, X).fit()
        print("\nMODEL 1: MULTIPLE LINEAR REGRESSION")
        print(m.summary())
        self.results['linear'] = m
        return m

    def elasticity(self):
        y = self.d['log_revenue']
        X = sm.add_constant(self.d[['log_ad_spend']])
        m = sm.OLS(y, X).fit()
        e = m.params['log_ad_spend']
        print("\nMODEL 2: ELASTICITY ANALYSIS")
        print(m.summary())
        print(f"Elasticity = {e:.3f} → {'Inelastic' if e < 1 else 'Elastic'} demand")
        self.results['elasticity'] = m
        return m

    def logistic_regression(self):
        y = self.data['Has_Revenue']
        X = sm.add_constant(self.data[['Ad_Spend_excl_VAT', 'CTR', 'Total_Ad_Clicks', 'CPC']])
        m = sm.Logit(y, X).fit(disp=0)
        auc = roc_auc_score(y, m.predict(X))
        print("\nMODEL 3: SUCCESS PROBABILITY (LOGIT)")
        print(m.summary())
        print(f"AUC = {auc:.3f}")
        self.results['logit'] = m
        return m

    def diminishing_returns(self):
        X = self.d[['Ad_Spend_excl_VAT']].values
        y = self.d['Total_Revenue_with_Ads'].values
        poly = PolynomialFeatures(2).fit_transform(X)
        m = LinearRegression().fit(poly, y)
        r2 = r2_score(y, m.predict(poly))
        rmse = np.sqrt(mean_squared_error(y, m.predict(poly)))
        quad = m.coef_[2]
        opt = -m.coef_[1] / (2 * quad) if quad < 0 else None
        print("\nMODEL 4: DIMINISHING RETURNS")
        print(
            f"R²={r2:.3f}, RMSE={rmse:.2f}, Intercept={m.intercept_:.2f}, "
            f"Linear={m.coef_[1]:.2f}, Quad={quad:.6f}"
        )
        if opt:
            print(f"Optimal spend ≈ {opt:.2f}€")
        self.results['poly'] = m
        return m

    def insights(self):
        print("\nBUSINESS INSIGHTS")
        avg_roi = self.data[self.data['ROAS'] > 0]['ROAS'].mean()
        success_rate = (self.data['Has_Revenue'] == 1).mean() * 100
        avg_cpc = self.data[self.data['CPC'] > 0]['CPC'].mean()
        print(f"ROI: {avg_roi:.2f}€/€ | Success Rate: {success_rate:.1f}% | Avg CPC: {avg_cpc:.2f}€")

if __name__ == "__main__":
    # Load the cleaned data (must have Ad_Spend_excl_VAT > 0 entries)
    df_clean = pd.read_csv("data/sample_campaigns.csv")
    an = SMEeBayAnalyzer(df_clean)
    an.linear_regression()
    an.elasticity()
    an.logistic_regression()
    an.diminishing_returns()
    an.insights()
