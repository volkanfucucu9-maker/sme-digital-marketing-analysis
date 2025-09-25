"""
Main analysis notebook functions
Extracted from Jupyter notebook for reproducibility
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression

def run_complete_analysis(df):
    """
    Run complete SME digital marketing analysis
    
    Parameters:
    df: DataFrame with campaign data
    
    Returns:
    dict: Results including models, metrics, and visualizations
    """
    
    # Clean data
    df_clean = clean_campaign_data(df)
    
    # Split data
    X = df_clean.drop('Total_Revenue_with_Ads', axis=1)
    y = df_clean['Total_Revenue_with_Ads']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train models
    results = {
        'ols_model': train_ols(X_train, y_train),
        'elasticity': calculate_elasticity(df_clean),
        'cv_scores': cross_validate(X, y),
        'test_r2': evaluate_on_test(X_test, y_test)
    }
    
    return results

def check_assumptions(model, X, y):
    """Check regression assumptions"""
    residuals = model.resid
    
    # Normality
    _, p_norm = stats.shapiro(residuals)
    
    # Homoscedasticity
    _, p_homo = het_breuschpagan(residuals, X)
    
    # Independence
    dw = durbin_watson(residuals)
    
    return {
        'normality_p': p_norm,
        'homoscedasticity_p': p_homo,
        'durbin_watson': dw
    }
