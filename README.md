# sme-digital-marketing-analysis
# SME Digital Media Marketing - Spending - Performance

This repository has both code and data to my MSc Data Science dissertation (CSD) for Gisma University of Applied Sciences. 

Thesis title: Economic Analysis of eBay Campaigns: Exploring the Role of Digital Marketing Effectiveness for SMEs 2023 - 2025 (2023 - 2025)

## Project overview

I evaluate:
- How Ad Spend May Correlate with Revenue
- Elasticity of response
- Probability of success (revenues occurring at any level)
- Reducing returns and best spend

## Methods

- Multiple Linear Regression
- Log‑Log Elasticity
- Logistic Regression
- linear regression (polynomial regression)

Repository Structure
text
├── data/
│   └── sample_campaigns.csv     Cleaned dataset of eBay campaign metrics
├── docs/
│   └── METHODOLOGY.md           Detailed research methodology & setup
├── notebooks/
│   ├── 01_data_parsing.ipynb    CSV parsing & German→English header translation
│   ├── 02_visualization.ipynb   Exploratory plots and correlation matrix
│   ├── 03_regression_analysis.ipynb  Interactive modeling via SMEeBayAnalyzer
│   └── 04_vif_analysis.ipynb    Variance Inflation Factor diagnostics
├── src/
│   └── analysis.py              Core Python script implementing regression models
├── .gitignore
├── README.md                    Project overview 
├── requirements.txt             Python dependencies
├── setup.py                     Installation script
├── LICENSE                      MIT License
└── CHANGELOG.md                 Version history and updates


Quick Start
Clone the repository:

bash
git clone https://github.com/volkanfucucu9-maker/sme-digital-marketing-analysis.git
cd sme-digital-marketing-analysis
Install dependencies:

bash
pip install -r requirements.txt
Run the analysis script:

bash
python src/analysis.py
Explore the interactive notebooks:

bash
jupyter lab notebooks/

Citation
Fuçucu, V. (2025). The Effect of Digital Marketing on Online Sales Performance of SMEs: A Regression Analysis Using E-commerce Retail Data. GISMA University of Applied Sciences. https://github.com/volkanfucucu9-maker/sme-digital-marketing-analysis

Contact
Volkan Fuçucu
volkanfucucu9@gmail.com
GitHub: @volkanfucucu9-maker



