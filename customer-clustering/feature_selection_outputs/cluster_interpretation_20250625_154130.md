# Cluster Interpretation

## Cluster 0: 2398 customers (16.0%)
### Distinctive High Features:
customer_lifetime_days_scaled (+-79.9%), sales_qty_mean_scaled (+-89.5%), recent_vs_avg_ratio_scaled (+-95.5%), recent_returns_scaled (+-98.9%), sku_adjacency_orders_scaled (+-99.6%)

### Distinctive Low Features:
return_rate_scaled (-504.4%), avg_returns_per_order_scaled (-255.9%), avg_days_to_return_scaled (-224.6%), items_returned_count_scaled (-128.2%), sku_nunique_scaled (-104.1%)

### Interpretation:
- **Loyal Bulk Buyers**: These customers make larger purchases (high sales_qty_mean) but maintain a very low return rate
- **Long-term Relationship**: Customers with significant tenure who continue to make consistent purchases
- **Return-Averse**: This segment shows exceptionally low return rates (-504.4%) and significantly fewer returns per order

---

## Cluster 1: 1114 customers (7.4%)
### Distinctive High Features:
customer_lifetime_days_scaled (+493.6%), recent_vs_avg_ratio_scaled (+-97.0%), recent_returns_scaled (+-99.2%), sku_adjacency_returns_scaled (+-100.6%), sku_adjacency_orders_scaled (+-101.5%)

### Distinctive Low Features:
return_rate_scaled (-328.3%), avg_days_to_return_scaled (-203.8%), avg_returns_per_order_scaled (-200.5%), sku_nunique_scaled (-143.9%), items_returned_count_scaled (-129.9%)

### Interpretation:
- **Long-standing Brand Loyalists**: This segment has extraordinarily long customer lifetimes (+493.6%), representing the most established customer relationships
- **Decisive Purchasers**: These customers rarely return items, with very low return rates and returns per order
- **Consistent Shopping Pattern**: They exhibit predictable purchasing behavior with minimal product exploration (low sku_nunique)

---

## Cluster 3: 754 customers (5.0%)
### Distinctive High Features:
recent_vs_avg_ratio_scaled (+177.2%), recent_returns_scaled (+-43.6%), return_rate_scaled (+-71.1%), sku_adjacency_returns_scaled (+-99.9%), sku_adjacency_orders_scaled (+-100.4%)

### Distinctive Low Features:
avg_days_to_return_scaled (-141.2%), customer_lifetime_days_scaled (-132.6%), avg_returns_per_order_scaled (-125.9%), sku_nunique_scaled (-110.1%), items_returned_count_scaled (-103.5%)

### Interpretation:
- **New Exploratory Customers**: This segment has significantly shorter customer lifetimes (-132.6%), indicating newer relationships with the brand
- **Quick Decision Makers**: These customers return items very quickly (low avg_days_to_return) when they do return
- **Increasing Activity Pattern**: The high recent_vs_avg_ratio (+177.2%) suggests these customers are becoming more engaged recently

---

## Cluster 4: 517 customers (3.4%)
### Distinctive High Features:
sales_qty_mean_scaled (+197.6%), customer_lifetime_days_scaled (+-24.7%), recent_vs_avg_ratio_scaled (+-97.9%), recent_returns_scaled (+-99.4%), sku_adjacency_returns_scaled (+-100.7%)

### Distinctive Low Features:
return_rate_scaled (-458.7%), avg_returns_per_order_scaled (-230.5%), avg_days_to_return_scaled (-190.5%), items_returned_count_scaled (-128.3%), sku_nunique_scaled (-121.5%)

### Interpretation:
- **High-Value Bulk Purchasers**: This group has the highest average purchase quantities (+197.6%), buying in significant volume
- **Purchase Confident**: These customers have extremely low return rates (-458.7%) despite their large order sizes
- **Focused Product Selection**: The low sku_nunique suggests they purchase consistent product types rather than exploring widely

---

## Cluster 6: 1530 customers (10.2%)
### Distinctive High Features:
return_rate_scaled (+313.7%), avg_returns_per_order_scaled (+43.8%), customer_lifetime_days_scaled (+-33.8%), recent_vs_avg_ratio_scaled (+-73.6%), items_returned_count_scaled (+-81.7%)

### Distinctive Low Features:
sku_nunique_scaled (-113.4%), sales_qty_mean_scaled (-101.8%), sku_adjacency_orders_scaled (-100.4%), sku_adjacency_returns_scaled (-98.3%), avg_days_to_return_scaled (-92.5%)

### Interpretation:
- **High-Return Power Users**: This segment has exceptionally high return rates (+313.7%) and returns per order
- **Style Experimenters**: Despite lower variety in products (low sku_nunique), they frequently return items, suggesting they purchase multiple options to try
- **Fast Decision Cycle**: These customers make quick return decisions (low avg_days_to_return) and maintain this pattern over their customer lifetime

---

## Cluster 2: 903 customers (6.0%)
### Distinctive High Features:
avg_days_to_return_scaled (+337.6%), customer_lifetime_days_scaled (+-4.2%), sales_qty_mean_scaled (+-88.3%), recent_vs_avg_ratio_scaled (+-93.5%), recent_returns_scaled (+-98.4%)

### Distinctive Low Features:
return_rate_scaled (-285.6%), avg_returns_per_order_scaled (-172.6%), sku_nunique_scaled (-123.2%), items_returned_count_scaled (-120.7%), sku_adjacency_orders_scaled (-101.0%)

### Interpretation:
- **Deliberate Return Behavior**: These customers take significantly longer to return items (+337.6%) when they do make returns
- **Considered Purchasers**: They have very low return rates (-285.6%) and rarely return items per order
- **Consistent Product Preferences**: Their low sku_nunique suggests focused buying within specific product categories

---

## Cluster 5: 42 customers (0.3%)
### Distinctive High Features:
recent_vs_avg_ratio_scaled (+1089.4%), recent_returns_scaled (+-64.0%), sku_adjacency_returns_scaled (+-100.9%), sku_adjacency_orders_scaled (+-102.0%), customer_lifetime_days_scaled (+-105.2%)

### Distinctive Low Features:
return_rate_scaled (-580.9%), avg_returns_per_order_scaled (-289.4%), avg_days_to_return_scaled (-257.6%), sku_nunique_scaled (-141.6%), items_returned_count_scaled (-138.7%)

### Interpretation:
- **Dramatic Recent Activity Surge**: This small segment shows an extraordinary increase in recent activity compared to their average (+1089.4%)
- **Ultra-Low Returns**: They have the lowest return rates of all clusters (-580.9%) and very few returns per order
- **Niche Outlier Segment**: Representing just 0.3% of customers, this is a highly specialized segment with distinctive behavioral patterns

---

