# Data Documentation

## Data Privacy Notice
Original eBay campaign data cannot be shared due to commercial confidentiality.

## 📊 Variable Descriptions

| Variable | Type | Description | Unit |
|----------|------|-------------|------|
| Ad_Spend_excl_VAT | Float | Advertising expenditure excluding VAT | EUR |
| Total_Ad_Clicks | Integer | Number of clicks on advertisements | Count |
| Ad_Impressions | Integer | Number of times ads were displayed | Count |
| Total_Revenue_with_Ads | Float | Revenue generated from advertised products | EUR |
| Units_Sold_with_Ads | Integer | Number of items sold | Count |
| CTR | Float | Click-through rate (derived) | Percentage |

Data Processing Pipeline
1. European format conversion (comma → period)
2. Currency symbol removal
3. Zero-revenue campaign exclusion (21 campaigns)
4. Outlier detection (3×IQR method)
5. Feature engineering (CTR calculation)

## Sample Statistics
- Original campaigns: ~150
- After cleaning: 99
- Training set: 79 (80%)
- Test set: 20 (20%)
