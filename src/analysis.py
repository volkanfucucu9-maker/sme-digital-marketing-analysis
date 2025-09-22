# src/analysis.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# ========== 1) DATA LOADER (ilk bu!) ==========
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

CANDIDATES = [
    DATA_DIR / "Promotion_Listing_AKF.csv",  
]

def _find_data():
    for p in CANDIDATES:
        if p.exists():
            print(f"[INFO] Using data file: {p.relative_to(ROOT)}")
            return p
    raise FileNotFoundError("No dataset found in /data. Put one of: "
                            + ", ".join(x.name for x in CANDIDATES))

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "Anzeigengebühren (ohne MwSt.)": "Ad_Spend_excl_VAT",
        "Anzeigen-Klicks insgesamt": "Total_Ad_Clicks",
        "Gesamtumsatz mit Anzeigen": "Total_Revenue_with_Ads",
        "Anzeigen-Impressions (über Platzierungen bei eBay)": "Impressions",
        "Rentabilität der Anzeigenkosten (Umsatz/Anzeigengebühren (ohne MwSt.))": "ROAS",
        "Ad Spend excl VAT": "Ad_Spend_excl_VAT",
        "Total Ad Clicks": "Total_Ad_Clicks",
        "Total Revenue with Ads": "Total_Revenue_with_Ads",
    }
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})

    if "CTR" not in df.columns and {"Total_Ad_Clicks","Impressions"}.issubset(df.columns):
        df["CTR"] = (df["Total_Ad_Clicks"] / df["Impressions"]).replace([np.inf, np.nan], 0).clip(0,1)
    if "CPC" not in df.columns and {"Ad_Spend_excl_VAT","Total_Ad_Clicks"}.issubset(df.columns):
        df["CPC"] = (df["Ad_Spend_excl_VAT"] / df["Total_Ad_Clicks"]).replace([np.inf, np.nan], 0)
    if "ROAS" not in df.columns and {"Total_Revenue_with_Ads","Ad_Spend_excl_VAT"}.issubset(df.columns):
        df["ROAS"] = (df["Total_Revenue_with_Ads"] / df["Ad_Spend_excl_VAT"]).replace([np.inf, np.nan], 0)
    if "Has_Revenue" not in df.columns and "Total_Revenue_with_Ads" in df.columns:
        df["Has_Revenue"] = (df["Total_Revenue_with_Ads"] > 0).astype(int)

    required = ["Ad_Spend_excl_VAT","Total_Revenue_with_Ads","Total_Ad_Clicks",
                "Impressions","CTR","CPC","ROAS","Has_Revenue"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns after normalisation: {miss}\nHave: {list(df.columns)}")
    return df

DATA_FILE = _find_data()
df_raw   = pd.read_csv(DATA_FILE)
df_clean = _normalise_columns(df_raw)

# ========== 2) (İSTEĞE BAĞLI) VIF — loader’dan SONRA ==========
d = df_clean[(df_clean['Ad_Spend_excl_VAT'] > 0) & (df_clean['Total_Ad_Clicks'] > 0)].copy()
X = d[['Ad_Spend_excl_VAT','Total_Ad_Clicks','CTR','CPC']].dropna()
Xc = add_constant(X, has_constant='add')
vif = pd.DataFrame({
    "Variable": Xc.columns,
    "VIF": [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]
})
print("\nVariance Inflation Factors:\n", vif[vif["Variable"] != "const"])

# ========== 3) MODELLER ==========
# ... (buradan sonra sınıfın/OLS/logit/quadratic fonksiyonların)
class SMEeBayAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.results = {}
        self.prepare_data()

    def prepare_data(self):
        d = self.data[(self.data['Ad_Spend_excl_VAT'] > 0) &
                      (self.data['Total_Ad_Clicks'] > 0)].copy()
        d['log_ad_spend'] = np.log(d['Ad_Spend_excl_VAT'])
        d['log_revenue']  = np.log(d['Total_Revenue_with_Ads'] + 1)
        self.d = d
        print(f"Analysis data prepared: {len(d)} campaigns with spend & clicks")

    def vif(self):
        X = add_constant(self.d[['Ad_Spend_excl_VAT','Total_Ad_Clicks','CTR','CPC']], has_constant='add')
        vif_df = pd.DataFrame({
            "Variable": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        })
        print("\nVariance Inflation Factors:\n", vif_df[vif_df["Variable"] != "const"])
        self.results['vif'] = vif_df
        return vif_df

    def linear_regression(self):
        y = self.d['Total_Revenue_with_Ads']
        X = add_constant(self.d[['Ad_Spend_excl_VAT','Total_Ad_Clicks','CTR','CPC']], has_constant='add')
        m = sm.OLS(y, X).fit().get_robustcov_results(cov_type="HC3")
        print("\nMODEL 1: OLS (HC3 robust SE)")
        print(m.summary())
        self.results['ols'] = m
        return m

    def elasticity(self):
        y = self.d['log_revenue']
        X = add_constant(self.d[['log_ad_spend']], has_constant='add')
        m = sm.OLS(y, X).fit().get_robustcov_results(cov_type="HC3")
        print("\nMODEL 2: Log–log elasticity (HC3)")
        print(m.summary())
        print(f"Elasticity (log_ad_spend → log_revenue) = {m.params['log_ad_spend']:.3f}")
        self.results['elasticity'] = m
        return m

    def logistic_regression(self):
        d = self.d.copy()
        if 'Has_Revenue' not in d.columns:
            d['Has_Revenue'] = (d['Total_Revenue_with_Ads'] > 0).astype(int)
        y = d['Has_Revenue']
        X = add_constant(d[['Ad_Spend_excl_VAT','CTR','Total_Ad_Clicks','CPC']], has_constant='add')
        m = sm.Logit(y, X).fit(disp=0).get_robustcov_results(cov_type="HC3")
        # AUC (guard)
        try:
            auc = roc_auc_score(y, m.model.predict(m.params))
        except Exception:
            auc = float('nan')
        print("\nMODEL 3: Logistic (HC3 robust SE)")
        print(m.summary())
        print(f"AUC = {auc if auc == auc else 'NA'}")
        self.results['logit'] = m
        return m

    def diminishing_returns(self):
        X = self.d[['Ad_Spend_excl_VAT']].values
        y = self.d['Total_Revenue_with_Ads'].values
        poly = PolynomialFeatures(2, include_bias=True).fit_transform(X)
        m = LinearRegression().fit(poly, y)
        yhat = m.predict(poly)
        r2 = r2_score(y, yhat)
        rmse = np.sqrt(mean_squared_error(y, yhat))
        beta1, beta2 = m.coef_[1], m.coef_[2]
        opt = -beta1/(2*beta2) if beta2 < 0 else None
        print("\nMODEL 4: Quadratic (in €)")
        print(f"R²={r2:.3f}, RMSE={rmse:.2f}, Linear={beta1:.2f}, Quad={beta2:.6f}")
        if opt is not None:
            print(f"Efficient spend band / optimum (indicative) ≈ €{opt:.2f}")
        self.results['quad'] = (m, r2, rmse, opt)
        return m

if __name__ == "__main__":
    # In Colab or locally, this must run BEFORE any VIF/model
    df_clean = pd.read_csv("Promotion_listing_AKF.csv")
    an = SMEeBayAnalyzer(df_clean)
    an.vif()
    an.linear_regression()
    an.elasticity()
    an.logistic_regression()
    an.diminishing_returns()
