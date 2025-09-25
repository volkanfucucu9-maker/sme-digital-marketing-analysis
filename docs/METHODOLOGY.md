# Detailed Methodology

## Research Design
This study employs quantitative regression analysis to examine the relationship between digital advertising expenditure and sales performance for SMEs operating on eBay marketplace. The analysis uses cross-sectional campaign-level data from 2023-2024.

## Model Specifications
- **Multiple Linear Regression**:
Revenue = β₀ + β₁(Ad_Spend) + β₂(Clicks) + β₃(Impressions) + β₄(CTR) + β₅(Units_Sold) + ε

- **Log-Log Elasticity Model**:
ln(Revenue) = γ₀ + γ₁ln(Ad_Spend) + ν
  
- **Simple Linear Model (Baseline)**:
Revenue = α₀ + α₁(Ad_Spend) + ε
## Data Preparation
1. **Data Loading**: CSV parsing with semicolon delimiter, skip first 2 rows
2. **Column Translation**: German headers to English (e.g., 'Anzeigengebühren (ohne MwSt.)' → 'Ad_Spend_excl_VAT')
3. **Numeric Conversion**: European format (comma decimals, period thousands) to standard
4. **Zero Revenue Removal**: Excluded campaigns with Total_Revenue_with_Ads ≤ 0
5. **Feature Engineering**: CTR = (Total_Ad_Clicks / Ad_Impressions) × 100
6. **Multicollinearity Handling**: Removed ROAS and CPC variables
7. **Outlier Detection**: 3×IQR method applied to all numeric columns
8. **Train-Test Split**: 80% training, 20% testing (random_state=42)

## Statistical Procedures
- **Estimation**: Ordinary Least Squares (OLS) via statsmodels
- **Validation**: 5-fold cross-validation
- **Assumption Testing**: 
  - Normality (Shapiro-Wilk test)
  - Homoscedasticity (Breusch-Pagan test)
  - Independence (Durbin-Watson test)
  - Multicollinearity (Variance Inflation Factor)

## Software Environment
- Python 3.10.12
- pandas==1.5.3
- numpy==1.23.5
- statsmodels==0.14.0
- scikit-learn==1.3.0
- matplotlib==3.7.1
- seaborn==0.12.2
- scipy==1.10.1

## Reproducibility
All code available at: https://github.com/volkanfucucu9-maker/sme-digital-marketing-analysis
- Clone repository
- Install dependencies: `pip install -r requirements.txt`
- Run analysis: `python src/analysis.py`

# Citation
Fuçucu, V. (2025). The Effect of Digital Marketing on Online Sales Performance of SMEs: A Regression Analysis Using E-commerce Retail Data. Master's Thesis, GISMA University of Applied Sciences. https://github.com/volkanfucucu9-maker/sme-digital-marketing-analysis

# Contact
**Volkan Fuçucu**  
Email: [contact via GitHub]  
GitHub: @volkanfucucu9-maker
