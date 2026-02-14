# clustering_credit_card_customers (Unsupervised Learning)

This project applies KMeans clustering to segment credit card customers based on their financial behavior and transaction patterns.

Unlike supervised learning, clustering identifies natural customer segments without predefined labels.

---

## Project Overview

- **Problem Type:** Unsupervised Learning (Clustering)
- **Algorithm:** KMeans
- **Optimal Clusters:** 5 (determined using Elbow Method)
- **Deployment:** Streamlit app on Hugging Face Spaces
- **Dataset Size:** 8,950 customers
- **Features:** 17 financial behavior variables

---

## Dataset

The dataset contains credit card customer behavior including:

- Balance
- Purchases (One-off & Installments)
- Cash Advance usage
- Purchase and Cash Advance frequencies
- Credit Limit
- Payments
- Minimum Payments
- Full Payment Percentage
- Tenure

The customer ID column was removed before modeling.

---

## Data Preprocessing

### Handling Missing Values
Missing values in:
- `CREDIT_LIMIT`
- `MINIMUM_PAYMENTS`

were imputed using median values.

### Log Transformation
Highly skewed financial variables were transformed using:


to reduce skewness and improve clustering stability.

### Feature Scaling
All features were standardized using `StandardScaler` to ensure equal contribution in distance-based clustering.

---

## Modeling

### Cluster Selection

- The silhouette score suggested k=2 (strong separation).
- The elbow method suggested k=5.
- For business relevance and finer segmentation, k=5 was selected.

### Final Model
KMeans with:
- `n_clusters=5`
- `random_state=42`
- `n_init=10`

---

## Cluster Interpretation

The model identified five distinct customer segments:

1. **Cash Advance Heavy Users**
   - High cash advance usage
   - Potential liquidity dependency

2. **Low Activity Conservative Users**
   - Low balance and purchases
   - Conservative financial behavior

3. **High Value Premium Customers**
   - High purchases and transaction frequency
   - Higher credit limits
   - Strong payment behavior

4. **Installment / Revolving Users**
   - Frequent installment purchases
   - Revolving credit usage patterns

5. **Balanced Mid-Level Customers**
   - Moderate activity across multiple features

---

## Deployment

Saved artifacts:

- `kmeans_credit_card.joblib`
- `scaler_credit_card.joblib`
- `feature_list.json`

The Streamlit app allows users to input financial behavior metrics and receive:

- Cluster ID
- Customer Segment Label
- Segment Description

---

## Conclusion

This project demonstrates how unsupervised learning can uncover hidden behavioral patterns in financial datasets.

Although silhouette analysis indicated a simpler 2-cluster structure, selecting 5 clusters enabled more granular and business-relevant segmentation.

Such segmentation can support:

- Risk assessment
- Targeted marketing
- Personalized credit offers
- Customer lifetime value optimization

---

## Future Improvements

- PCA-based dimensionality reduction before clustering
- Silhouette score comparison across multiple k values
- Hierarchical clustering comparison
- Cluster stability analysis
