import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# --- SMART LOADER for Promotion_Listing_AKF.csv (put right after imports) ---
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "Promotion_Listing_AKF.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

def try_read(path):
    """Try several common eBay export formats; return first successful DataFrame."""
    trials = [
        dict(sep=";", decimal=",", thousands=".", encoding="utf-8-sig", engine="python"),
        dict(sep=";", decimal=",", thousands=".", encoding="utf-8",     engine="python"),
        dict(sep=",", decimal=".", thousands=",", encoding="utf-8-sig", engine="python"),
        dict(sep=",", decimal=".", thousands=",", encoding="utf-8",     engine="python"),
        dict(sep="\t", decimal=",", thousands=".", encoding="utf-8-sig", engine="python"),
    ]
    last_err = None
    for opts in trials:
        try:
            df = pd.read_csv(path, **opts)
            if df.shape[1] >= 6:  # at least some columns
                return df
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("Could not parse CSV with common settings.")

raw = try_read(DATA_FILE)

# Map German/English headers to your analysis schema
colmap = {
    "Anzeigengebühren (ohne MwSt.)": "Ad_Spend_excl_VAT",
    "Anzeigen-Klicks insgesamt": "Total_Ad_Clicks",
    "Gesamtumsatz mit Anzeigen": "Total_Revenue_with_Ads",
    "Anzeigen-Impressions (über Platzierungen bei eBay)": "Impressions",
    "Rentabilität der Anzeigenkosten (Umsatz/Anzeigengebühren (ohne MwSt.))": "ROAS",
    # English variants (if already exported in EN)
    "Ad Spend excl VAT": "Ad_Spend_excl_VAT",
    "Total Ad Clicks": "Total_Ad_Clicks",
    "Total Revenue with Ads": "Total_Revenue_with_Ads",
    "Impressions": "Impressions",
    "ROAS": "ROAS",
}
raw = raw.rename(columns={k: v for k, v in colmap.items() if k in raw.columns})

# Derive missing fields safely
if "CTR" not in raw.columns and {"Total_Ad_Clicks","Impressions"}.issubset(raw.columns):
    raw["CTR"] = (raw["Total_Ad_Clicks"] / raw["Impressions"]).replace([np.inf, np.nan], 0).clip(0,1)
if "CPC" not in raw.columns and {"Ad_Spend_excl_VAT","Total_Ad_Clicks"}.issubset(raw.columns):
    raw["CPC"] = (raw["Ad_Spend_excl_VAT"] / raw["Total_Ad_Clicks"]).replace([np.inf, np.nan], 0)
if "ROAS" not in raw.columns and {"Total_Revenue_with_Ads","Ad_Spend_excl_VAT"}.issubset(raw.columns):
    raw["ROAS"] = (raw["Total_Revenue_with_Ads"] / raw["Ad_Spend_excl_VAT"]).replace([np.inf, np.nan], 0)
if "Has_Revenue" not in raw.columns and "Total_Revenue_with_Ads" in raw.columns:
    raw["Has_Revenue"] = (raw["Total_Revenue_with_Ads"] > 0).astype(int)

required = ["Ad_Spend_excl_VAT","Total_Revenue_with_Ads","Total_Ad_Clicks",
            "Impressions","CTR","CPC","ROAS","Has_Revenue"]
missing = [c for c in required if c not in raw.columns]
if missing:
    # Yardımcı: ilk satırları yazdır ki başlık doğru mu görelim
    print("[DEBUG] Header preview:\n", raw.head(3).to_string())
    raise ValueError(f"Required columns missing after normalisation: {missing}\nHave: {list(raw.columns)}")

df_clean = raw.copy()
print(f"[INFO] Loaded '{DATA_FILE.name}' with shape {df_clean.shape}")
# --- END SMART LOADER ---


# ========== 2) (İSTEĞE BAĞLI) VIF — loader’dan SONRA ==========
d = df_clean[(df_clean['Ad_Spend_excl_VAT'] > 0) & (df_clean['Total_Ad_Clicks'] > 0)].copy()
X = d[['Ad_Spend_excl_VAT','Total_Ad_Clicks','CTR','CPC']].dropna()
Xc = sm.add_constant(X, has_constant='add')
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
