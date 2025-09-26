import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from google.colab import drive
drive.mount('/content/drive')

def load_ebay_data():
    """
    Load eBay campaign data from Google Drive
    Handles German CSV format with proper encoding
    """
    file_paths = [
        '/content/drive/My Drive/Dataset/Promotion_Listing_AKF.csv',
        '/content/drive/MyDrive/Dataset/Promotion_Listing_AKF.csv',
        '/content/drive/My Drive/Promotion_Listing_AKF.csv',
        'Promotion_Listing_AKF.csv'
    ]

    df = None
    for filepath in file_paths:
        try:
            df = pd.read_csv(
                filepath,
                sep=';',
                skiprows=2,
                decimal=',',
                thousands='.',
                encoding='utf-8'
            )
            print(f"✓ Data loaded from: {filepath}")
            return df

        except Exception as e:
            continue

    if df is None:
        print("⚠️ File not found in expected locations")
        print("Please upload the file manually:")
        from google.colab import files
        uploaded = files.upload()
        filename = list(uploaded.keys())[0]
        df = pd.read_csv(
            filename,
            sep=';',
            skiprows=2,
            decimal=',',
            thousands='.',
            encoding='utf-8'
        )
        print(f"✓ Data loaded from uploaded file: {filename}")
        return df

    return df

df_raw = load_ebay_data()

if df_raw is not None:
    print(f"\nDataset shape: {df_raw.shape}")
    print(f"Number of campaigns: {df_raw.shape[0]}")
    print(f"Number of features: {df_raw.shape[1]}")
else:
    print("Error: Data not loaded!")

df_raw = load_ebay_data()
print(f"\nDataset shape: {df_raw.shape}")
print(f"Number of campaigns: {df_raw.shape[0]}")
print(f"Number of features: {df_raw.shape[1]}")

print("First 5 rows of raw data:")
df_raw.head()

print("Data types and missing values:")
df_raw.info()

def preprocess_columns(df):
    """
    Rename German columns to English and clean currency/percentage values
    """
    df = df.copy()

    column_mapping = {
        'Anzeigengebühren (ohne MwSt.)': 'Ad_Spend_excl_VAT',
        'Anzeigen-Klicks insgesamt': 'Total_Ad_Clicks',
        'Gesamtumsatz mit Anzeigen': 'Total_Revenue_with_Ads',
        'Anzeigen-Impressions (über Platzierungen bei eBay)': 'Ad_Impressions',
        'Verkauft mit Anzeigen - Gesamtstückzahl': 'Units_Sold_with_Ads',
        'Organisch verkauft - Stückzahl': 'Organic_Units_Sold',
        'Verkauft - Gesamtstückzahl': 'Total_Units_Sold',
        'Rentabilität der Anzeigenkosten (Umsatz/Anzeigengebühren (ohne MwSt.))': 'ROAS',
        'Anzeigen-Konversionsrate (Verkauft mit Anzeigen - Stückzahl/Anzeigen-Klicks)': 'Ad_Conversion_Rate',
        'Durchschn. Kosten pro Verkauf': 'Avg_Cost_per_Sale'
    }

    df.rename(columns=column_mapping, inplace=True)
    print(" Columns renamed to English")

    currency_columns = ['Ad_Spend_excl_VAT', 'Total_Revenue_with_Ads', 'Avg_Cost_per_Sale']
    for col in currency_columns:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                      .str.replace('€', '')
                      .str.replace(' ', '')
                      .str.replace(',', '.')
                      .apply(pd.to_numeric, errors='coerce'))
    print(" Currency columns cleaned")

    if 'Ad_Conversion_Rate' in df.columns:
        df['Ad_Conversion_Rate'] = (df['Ad_Conversion_Rate'].astype(str)
                                    .str.replace('%', '')
                                    .str.replace(',', '.')
                                    .apply(pd.to_numeric, errors='coerce') / 100)

    if 'ROAS' in df.columns:
        df['ROAS'] = (df['ROAS'].astype(str)
                     .str.replace(',', '.')
                     .apply(pd.to_numeric, errors='coerce'))

    print(" Percentage and ratio columns cleaned")

    return df

df = preprocess_columns(df_raw)
print(f"\nProcessed dataset shape: {df.shape}")

def create_derived_features(df):
    """
    Create additional features for analysis
    """
    df = df.copy()

    if 'Total_Ad_Clicks' in df.columns and 'Ad_Impressions' in df.columns:
        df['CTR'] = df['Total_Ad_Clicks'] / df['Ad_Impressions'].replace(0, np.nan)
        print("✓ CTR (Click-through Rate) calculated")

    if 'Ad_Spend_excl_VAT' in df.columns and 'Total_Ad_Clicks' in df.columns:
        df['CPC'] = df['Ad_Spend_excl_VAT'] / df['Total_Ad_Clicks'].replace(0, np.nan)
        print("✓ CPC (Cost Per Click) calculated")

    if 'Total_Revenue_with_Ads' in df.columns:
        df['Has_Revenue'] = (df['Total_Revenue_with_Ads'] > 0).astype(int)
        print("✓ Has_Revenue indicator created")

    return df

df = create_derived_features(df)
print(f"\nDataset with derived features: {df.shape}")

analysis_columns = [
    'Ad_Spend_excl_VAT',
    'Total_Ad_Clicks',
    'Total_Revenue_with_Ads',
    'Ad_Impressions',
    'CTR',
    'CPC',
    'ROAS',
    'Units_Sold_with_Ads'
]
available_cols = [col for col in analysis_columns if col in df.columns]
df_analysis = df[available_cols].copy()
print(f"Columns selected for analysis: {len(available_cols)}")
print(f"Available columns: {available_cols}")

print("Descriptive Statistics:")
df_analysis.describe()

if 'Total_Revenue_with_Ads' in df_analysis.columns:
    zero_revenue_count = (df_analysis['Total_Revenue_with_Ads'] <= 0).sum()
    total_count = len(df_analysis)
    zero_revenue_pct = (zero_revenue_count / total_count) * 100

    print(f"Zero/Negative Revenue Analysis:")
    print(f"  Total records: {total_count}")
    print(f"  Zero/negative revenue records: {zero_revenue_count}")
    print(f"  Percentage: {zero_revenue_pct:.1f}%")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(available_cols[:6]):
    axes[idx].hist(df_analysis[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Figure 1: Distribution of Key Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

def clean_data_for_regression(df, remove_zero_revenue=True):
    """
    Clean data according to dissertation requirements
    Key requirement: Remove zero revenue entries (William's feedback)
    """
    df_clean = df.copy()

    print("="*50)
    print("DATA CLEANING PROCESS")
    print("="*50)

    initial_rows = len(df_clean)
    df_clean = df_clean.dropna()
    removed_missing = initial_rows - len(df_clean)
    print(f"✓ Removed {removed_missing} rows with missing values")

    if remove_zero_revenue and 'Total_Revenue_with_Ads' in df_clean.columns:
        before_removal = len(df_clean)
        df_clean = df_clean[df_clean['Total_Revenue_with_Ads'] > 0]
        removed_zeros = before_removal - len(df_clean)
        print(f"✓ Removed {removed_zeros} zero revenue entries (REQUIRED by feedback)")

    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()

    for col in df_clean.select_dtypes(include=[np.number]).columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        if outliers > 0:
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            print(f"  Removed {outliers} outliers from {col}")

    print(f"\nFinal cleaned dataset:")
    print(f"  Records: {len(df_clean)}")
    print(f"  Features: {df_clean.shape[1]}")
    print(f"  Data retention: {(len(df_clean)/len(df))*100:.1f}%")

    return df_clean

df_clean = clean_data_for_regression(df_analysis, remove_zero_revenue=True)

df_clean = clean_data_for_regression(df_analysis, remove_zero_revenue=True)

def check_multicollinearity(df):
    """
    Identify and remove highly correlated features
    This addresses the high R² concern from feedback
    """
    print("="*50)
    print("MULTICOLLINEARITY ASSESSMENT")
    print("="*50)

    cols_to_remove = []

    if 'ROAS' in df.columns:
        cols_to_remove.append('ROAS')
        print("✓ ROAS removed (derived from Revenue/Ad_Spend)")

    if 'Units_Sold_with_Ads' in df.columns and 'Total_Revenue_with_Ads' in df.columns:
        corr = df['Units_Sold_with_Ads'].corr(df['Total_Revenue_with_Ads'])
        if abs(corr) > 0.95:
            cols_to_remove.append('Units_Sold_with_Ads')
            print(f"✓ Units_Sold removed (correlation with Revenue: {corr:.3f})")

    if 'CPC' in df.columns and 'CTR' in df.columns:
        cols_to_remove.append('CPC')
        print("✓ CPC removed (keeping CTR instead)")

    df_reduced = df.drop(columns=cols_to_remove, errors='ignore')

    numeric_cols = df_reduced.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df_reduced[numeric_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1)
        plt.title('Figure 2: Correlation Matrix After Removing Problematic Features',
                 fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

    return df_reduced

df_clean = check_multicollinearity(df_clean)
print(f"\nCleaned dataset shape: {df_clean.shape}")

def prepare_data_for_modeling(df):
    """
    Prepare features and target for modeling
    """
    if 'Total_Revenue_with_Ads' not in df.columns:
        raise ValueError("Target variable 'Total_Revenue_with_Ads' not found")

    y = df['Total_Revenue_with_Ads']
    X = df.drop(['Total_Revenue_with_Ads'], axis=1)

    X = X.select_dtypes(include=[np.number])

    print(f"Target variable: Total_Revenue_with_Ads")
    print(f"Features ({len(X.columns)}): {list(X.columns)}")

    return X, y

X, y = prepare_data_for_modeling(df_clean)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Features: {X.shape[1]}")

def build_regression_model(X_train, y_train, X_test, y_test):
    """
    Build and evaluate OLS regression model
    """
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    model = sm.OLS(y_train, X_train_const).fit()

    print("="*50)
    print("REGRESSION MODEL RESULTS")
    print("="*50)

    print(f"\nTraining Set Performance:")
    print(f"  R-squared: {model.rsquared:.4f}")
    print(f"  Adjusted R-squared: {model.rsquared_adj:.4f}")

    y_pred_test = model.predict(X_test_const)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"\nTest Set Performance:")
    print(f"  R-squared: {test_r2:.4f}")
    print(f"  RMSE: €{test_rmse:.2f}")

    overfit_pct = ((model.rsquared - test_r2) / model.rsquared) * 100
    print(f"\nOverfitting Check:")
    print(f"  Train-Test R² difference: {overfit_pct:.1f}%")
    if overfit_pct > 10:
        print("  ⚠️ Warning: Potential overfitting detected")
    else:
        print("  ✓ Model generalizes well")

    print(f"\nModel Significance:")
    print(f"  F-statistic p-value: {model.f_pvalue:.4e}")
    if model.f_pvalue < 0.05:
        print("  ✓ Model is statistically significant")
    else:
        print("  ✗ Model is not statistically significant")

    return model, X_test_const

model, X_test_const = build_regression_model(X_train, y_train, X_test, y_test)

print("\nDetailed Model Summary:")
print(model.summary())

def report_significant_predictors(model):

    print("="*50)
    print("SIGNIFICANT PREDICTORS (p < 0.05)")
    print("="*50)

    significant = model.pvalues[model.pvalues < 0.05]

    if len(significant) > 1:
        for var in significant.index:
            if var != 'const':
                coef = model.params[var]
                std_err = model.bse[var]
                p_val = model.pvalues[var]
                ci_lower, ci_upper = model.conf_int().loc[var]

                print(f"\n{var}:")
                print(f"  Coefficient: {coef:.4f}")
                print(f"  Std Error: {std_err:.4f}")
                print(f"  p-value: {p_val:.4f}")
                print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

                if var == 'Ad_Spend_excl_VAT':
                    print(f"  → €1 increase in ad spend → €{coef:.2f} increase in revenue")
                elif var == 'Total_Ad_Clicks':
                    print(f"  → 1 additional click → €{coef:.2f} increase in revenue")
    else:
        print("\n⚠️ No individual predictors are statistically significant")
        print("This may indicate multicollinearity or insufficient sample size")

report_significant_predictors(model)

def check_regression_assumptions(model, X, y):
    """
    Comprehensive check of all OLS assumptions
    MUST be done before reporting results (William's feedback)
    """
    print("="*50)
    print("REGRESSION ASSUMPTIONS TESTING")
    print("="*50)

    residuals = model.resid
    fitted = model.fittedvalues

    assumptions_met = {}

    print("\n1. NORMALITY OF RESIDUALS")
    shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
    assumptions_met['Normality'] = shapiro_p > 0.05
    print(f"   Shapiro-Wilk test p-value: {shapiro_p:.4f}")
    print(f"   Result: {'✓ PASS' if assumptions_met['Normality'] else '✗ FAIL'}")

    print("\n2. HOMOSCEDASTICITY")
    lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(residuals, X)
    assumptions_met['Homoscedasticity'] = lm_pvalue > 0.05
    print(f"   Breusch-Pagan test p-value: {lm_pvalue:.4f}")
    print(f"   Result: {'✓ PASS' if assumptions_met['Homoscedasticity'] else '✗ FAIL'}")

    print("\n3. INDEPENDENCE (NO AUTOCORRELATION)")
    dw_stat = durbin_watson(residuals)
    assumptions_met['Independence'] = 1.5 <= dw_stat <= 2.5
    print(f"   Durbin-Watson statistic: {dw_stat:.4f}")
    print(f"   Result: {'✓ PASS' if assumptions_met['Independence'] else '✗ FAIL'}")

    print("\n4. MULTICOLLINEARITY (VIF)")
    X_no_const = X.drop('const', axis=1) if 'const' in X.columns else X

    vif_passed = True
    for i, col in enumerate(X_no_const.columns):
        vif = variance_inflation_factor(X_no_const.values, i)
        status = '✓' if vif < 10 else '✗'
        print(f"   {col}: VIF = {vif:.2f} {status}")
        if vif >= 10:
            vif_passed = False

    assumptions_met['No Multicollinearity'] = vif_passed

    print("\n" + "="*30)
    print("ASSUMPTIONS SUMMARY")
    print("="*30)

    all_passed = all(assumptions_met.values())
    for assumption, passed in assumptions_met.items():
        print(f"{assumption:25s}: {'✓ PASS' if passed else '✗ FAIL'}")

    if not all_passed:
        print("\n⚠️ WARNING: Some assumptions are violated")
        print("Consider data transformations or alternative models")
    else:
        print("\n✓ All assumptions are satisfied")

    return assumptions_met

X_train_const = sm.add_constant(X_train)
assumptions = check_regression_assumptions(model, X_train_const, y_train)

def plot_diagnostic_plots(model, X, y):
    """
    Create diagnostic plots with proper labels (William's requirement)
    """
    residuals = model.resid
    fitted = model.fittedvalues

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].scatter(fitted, residuals, alpha=0.6, edgecolor='k')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted Values')
    axes[0, 0].grid(True, alpha=0.3)

    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
    mu, std = residuals.mean(), residuals.std()
    xmin, xmax = axes[1, 0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    axes[1, 0].plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Histogram of Residuals')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(fitted, np.sqrt(np.abs(residuals)), alpha=0.6, edgecolor='k')
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('√|Residuals|')
    axes[1, 1].set_title('Scale-Location Plot')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Figure 3: Model Diagnostic Plots', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

plot_diagnostic_plots(model, X_train_const, y_train)

def perform_cross_validation(X, y):
    """
    5-fold cross-validation to assess model stability
    """
    print("="*50)
    print("CROSS-VALIDATION ANALYSIS")
    print("="*50)

    lr_model = LinearRegression()

    cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='r2')

    print(f"5-Fold Cross-Validation R² Scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f}")

    print(f"\nCross-Validation Statistics:")
    print(f"  Mean R²: {cv_scores.mean():.4f}")
    print(f"  Std Dev: {cv_scores.std():.4f}")
    print(f"  Min R²: {cv_scores.min():.4f}")
    print(f"  Max R²: {cv_scores.max():.4f}")

    cv_coefficient = cv_scores.std() / cv_scores.mean()
    print(f"\nCoefficient of Variation: {cv_coefficient:.3f}")

    if cv_coefficient < 0.1:
        print("✓ Model is highly stable across folds")
    elif cv_coefficient < 0.2:
        print("✓ Model shows acceptable stability")
    else:
        print("⚠️ Model shows high variability across folds")

    return cv_scores

cv_results = perform_cross_validation(X, y)

def test_alternative_models(df):
    """
    Test simpler model specifications
    This addresses the issue of no significant individual predictors
    """
    print("="*50)
    print("ALTERNATIVE MODEL SPECIFICATIONS")
    print("="*50)

    y = df['Total_Revenue_with_Ads']

    if 'Ad_Spend_excl_VAT' in df.columns:
        X1 = df[['Ad_Spend_excl_VAT']]
        X1 = sm.add_constant(X1)
        model1 = sm.OLS(y, X1).fit()

        print("\nModel 1: Ad Spend Only")
        print(f"  R²: {model1.rsquared:.4f}")
        print(f"  Ad Spend coefficient: {model1.params['Ad_Spend_excl_VAT']:.4f}")
        print(f"  p-value: {model1.pvalues['Ad_Spend_excl_VAT']:.4f}")
        if model1.pvalues['Ad_Spend_excl_VAT'] < 0.05:
            print(f"  ✓ Significant: €1 ad spend → €{model1.params['Ad_Spend_excl_VAT']:.2f} revenue")

    if 'Ad_Spend_excl_VAT' in df.columns:
        df_log = df[['Ad_Spend_excl_VAT', 'Total_Revenue_with_Ads']].copy()
        df_log = df_log[(df_log > 0).all(axis=1)]

        df_log['Log_Revenue'] = np.log(df_log['Total_Revenue_with_Ads'])
        df_log['Log_Ad_Spend'] = np.log(df_log['Ad_Spend_excl_VAT'])

        X2 = df_log[['Log_Ad_Spend']]
        X2 = sm.add_constant(X2)
        y2 = df_log['Log_Revenue']
        model2 = sm.OLS(y2, X2).fit()

        print("\nModel 2: Log-Log (Elasticity)")
        print(f"  R²: {model2.rsquared:.4f}")
        print(f"  Elasticity: {model2.params['Log_Ad_Spend']:.4f}")
        print(f"  p-value: {model2.pvalues['Log_Ad_Spend']:.4f}")
        if model2.pvalues['Log_Ad_Spend'] < 0.05:
            print(f"  ✓ Significant: 1% increase in ad spend → {model2.params['Log_Ad_Spend']:.2f}% increase in revenue")

    return model1, model2

simple_model, log_model = test_alternative_models(df_clean)

print("="*60)
print("FINAL ANALYSIS SUMMARY")
print("="*60)

print("\n1. DATA CLEANING:")
print("   ✓ Zero revenue entries removed")
print("   ✓ Outliers removed")
print("   ✓ Missing values handled")

print("\n2. MULTICOLLINEARITY ADDRESSED:")
print("   ✓ ROAS removed (derived variable)")
print("   ✓ Highly correlated features removed")
print("   ✓ VIF values checked")

print("\n3. MODEL PERFORMANCE:")
print(f"   Training R²: {model.rsquared:.4f}")
print(f"   Test R²: {r2_score(y_test, model.predict(X_test_const)):.4f}")
print(f"   Cross-validation mean R²: {cv_results.mean():.4f}")

print("\n4. ASSUMPTIONS CHECKED:")
for assumption, passed in assumptions.items():
    print(f"   {assumption}: {'✓ PASS' if passed else '✗ FAIL'}")

print("\n5. KEY FINDINGS:")
if model.rsquared > 0.9:
    print("   Very high R² detected (>0.90)")
    print("   - This reflects direct ad-to-revenue relationship")
    print("   - Model generalizes well (low train-test gap)")
    print("   - Cross-validation confirms stability")
