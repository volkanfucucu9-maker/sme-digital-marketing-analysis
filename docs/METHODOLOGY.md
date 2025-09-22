# Detailed Methodology

## Research Design
This study uses quantitative regression analysis to assess SME digital marketing effectiveness...
(Include model equations, data cleaning steps, environment setup, reproducibility instructions.)

## Model Specifications
- Multiple Linear Regression: \(Revenue = \beta_0 + \beta_1 AdSpend + \beta_2 Clicks + \beta_3 CTR + \beta_4 CPC + \varepsilon\)
- Elasticity Analysis (Log-Log): \(\ln(Revenue+1) = \alpha_0 + \alpha_1 \ln(AdSpend) + \mu\)
- Logistic Regression: \(P(\text{HasRevenue}=1) = \sigma(\gamma_0 + \gamma_1 AdSpend + \gamma_2 CTR + \dots)\)
- Diminishing Returns (Polynomial): \(Revenue = \delta_0 + \delta_1 AdSpend + \delta_2 AdSpend^2 + \nu\)

## Data Preparation
1. CSV parsing and German→English translation
2. Type conversions (commas → dots)
3. Filtering: Ad_Spend_excl_VAT > 0 and Total_Ad_Clicks > 0

## Software Environment
- Python 3.8+
- pandas, numpy, statsmodels, scikit-learn versions as in requirements.txt

Citation
Fuçucu, V. (2025). The Effect of Digital Marketing on Online Sales Performance of SMEs: A Regression Analysis Using E-commerce Retail Data (2023–2025). GISMA University of Applied Sciences. https://github.com/volkanfucucu9-maker/sme-digital-marketing-analysis


Contact
Volkan Fuçucu
volkanfucucu9@gmail.com
GitHub: @volkanfucucu9-maker

