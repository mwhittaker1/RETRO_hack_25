"""
Customer Return Clustering Features - Categorized

# Feature Selection and Optimization

The clustering pipeline implements multi-stage feature selection to improve model quality:

## Basic Feature Quality Controls
- **100% Null Features**: Features with 100% null values are automatically removed
- **Highly Correlated Features**: Features with correlation > 0.8 are reduced to avoid multicollinearity
- **Feature Selection Metadata**: Each run tracks which features were removed and why

## Advanced Feature Selection Techniques
The pipeline now includes advanced feature selection methods:
- **Variance Inflation Factor (VIF)**: Detects and removes multicollinearity beyond simple correlation
- **Low-Variance Feature Filtering**: Removes features with minimal variance (< 0.01)
- **Feature Importance Analysis**: Uses Random Forest to identify and remove low-importance features
- **PCA Component Analysis**: Examines principal component loadings to reduce dimensionality
- **Feature Stability Analysis**: Identifies unstable features across K-fold cross-validation
- **Information Value (IV) Analysis**: Measures predictive power of features for business outcomes

## Optimized Feature Selection Strategy
Features flagged by multiple methods are prioritized for removal:
- Primary candidates: Features flagged by 2+ selection techniques
- Secondary candidates: Features flagged by only one technique but with extreme values
- Gold layer tables track the feature selection methodology and results

## Correlation Groups Identified
The following feature groups show high correlation and are candidates for reduction:
- Return Rate Group: `RETURN_RATE`, `RETURN_RATIO`, `RETURN_FREQUENCY_RATIO`, `HIGH_RETURN_CATEGORY_AFFINITY`
- Order Volume Group: `SALES_ORDER_NO_nunique`, `RECENT_ORDERS`
- Return Volume Group: `ITEMS_RETURNED_COUNT`, `RETURN_PRODUCT_VARIETY`

# Feature Categories

📊 BASIC VOLUME METRICS
Core counting and frequency measures

SALES_ORDER_NO_nunique – Number of unique orders placed (order frequency)
SKU_nunique – Number of unique products purchased (product variety)
ITEMS_RETURNED_COUNT – Total number of items returned
SALES_QTY_mean – Average purchase quantity per item
AVG_ORDER_SIZE – Average order size (items per order)

🔄 RETURN BEHAVIOR PATTERNS
How customers return items

RETURN_RATE – Items returned / total items purchased (frequency of returns)
RETURN_RATIO – Quantity returned / quantity purchased (intensity of returns)
RETURN_PRODUCT_VARIETY – Number of different SKUs returned (breadth of return behavior)
AVG_RETURNS_PER_ORDER – Average items returned per order (batch return behavior)
RETURN_FREQUENCY_RATIO – Returns per order ratio
RETURN_INTENSITY – Return quantity / sales quantity per returned item (partial vs full return)
CONSECUTIVE_RETURNS – Count of orders with consecutive returns
AVG_CONSECUTIVE_RETURNS – Average number of consecutive returns for user lifetime

⏰ TEMPORAL & TIMING PATTERNS
When and how quickly customers act

CUSTOMER_LIFETIME_DAYS – Days between first and last order (customer tenure)
AVG_DAYS_TO_RETURN – Average days between order and return (return timing)
RETURN_TIMING_SPREAD – Variability in return timing
CUSTOMER_TENURE_STAGE – New, Growing, Mature (90 days, 180 days, 365 days)

📈 TREND & RECENCY ANALYSIS
Recent behavior vs. historical patterns

RECENT_ORDERS – Unique orders in last 90 days (recent purchase activity)
RECENT_RETURNS – Items returned in last 90 days (recent return activity)
RECENT_VS_AVG_RATIO – Recent return rate / historical return rate (trend in return behavior) *
ORDER_FREQUENCY_TREND – Trend in order frequency compared to historical average *
RETURN_FREQUENCY_TREND – Trend in return frequency compared to historical average *
BEHAVIOR_STABILITY_SCORE – 0-1 score of recent behavior compared to historical *

💰 MONETARY VALUE PATTERNS
Financial impact and value-based behavior

AVG_ORDER_VALUE – Average order value *
AVG_RETURN_VALUE – Average return value *
HIGH_VALUE_RETURN_AFFINITY – Z-score of high-value returns compared to average *

🏷️ PRODUCT & CATEGORY INTELLIGENCE
What customers buy and return

PRODUCT_CATEGORY_LOYALTY – Sum of (Category_purchases²) / (Total_purchases²) *
CATEGORY_DIVERSIRY_SCORE – Z-score of category purchase variety *
CAETEGORY_LOYALTY_SCORE – Z-score of category purchase loyalty *
HIGH_RETURN_CATEGORY_AFFINITY – Z-score of return rate by category *
HIGH_RISK_PRODUCT_AFFINITY – Customers trend to purchase high-risk return products *
HIGH_RISK_RETURN_AFFINITY – Customers trend to return high-risk return products *

🔗 ADJACENCY & REPEAT BEHAVIOR
Patterns in related purchases and returns

SKU_ADJACENCY_ORDERS – Number of orders with adjacent SKUs purchased ★
SKU_ADJACENCY_RETURNS – Number of orders with adjacent SKUs returned ★
SKU_ADJACENCY_TIMING – Order adjacency timing (time between adjacent SKU orders) ★
SKU_ADJACENCY_RETURN_TIMING – Return adjacency timing (time between adjacent SKU returns) ★

🌊 SEASONAL & TREND SUSCEPTIBILITY
Response to external patterns and trends

SEASONAL_SUSCEPTIBIITY_RETURNS – Customer's susceptibility to seasonal trends in returns *
SEASONAL_SUSCEPTIBIITY_ORDERS – Customer's susceptibility to seasonal trends in orders *
TREND_PRODUCT_CATEGORY_RETURN_RATE – Compares returned product category to recency of 
        other customer returns (same product category within 90 days) *
TREND_SKU_RETURN_RATE – Compares returned SKU to recency of other customer returns 
        (same SKU within 90 days) *
TREND_PRODUCT_CATEGORY_ORDER_RATE – Compares ordered product category to recency of 
        other customer orders *
TREND_SKU_ORDER_RATE – Compares ordered SKU to recency of other customer orders *

"""

