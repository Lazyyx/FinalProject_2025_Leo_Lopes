# KuaiRec Recommender System Project

This project was developed as part of the *Recommender Systems* course and centers around the **KuaiRec dataset**, a large-scale, fully-observed dataset from the Kuaishou short-video platform. The goal is to explore, preprocess, and model real-world user-video interactions, and evaluate the effectiveness of several recommendation algorithms.

---

## **Overview**

I implemented multiple stages in this project:
1. **Data Analysis & Visualization**
2. **Data Cleaning and Preprocessing**
3. **Baseline Recommenders**:
   - Popularity-Based
   - Matrix Factorization (SVD)
4. **Alternating Least Squares (ALS)**
5. **Deep Autoencoder**
6. **Evaluation and Interpretation**

---

## **1. Data Analysis & Visualization**

I began my exploration in `EDA.ipynb`, a notebook that was already given, which included:
- Distribution of watch ratios
- User/video interaction counts
- Initial sanity checks

I extended this analysis in `EDA_advanced.ipynb` with:
- Statistics on `social_network`, `item_categories` etc.
- Time-series plots of daily activity
- Video duration vs. average watch ratio
- Watch ratio distribution across top-5 video tags

**Key Observations**:
- Most users have only one friend
- Most users watch a small number of videos intensively.
- Shorter videos are more likely to be rewatched (`watch_ratio ≥ 2`).
- A few tags (e.g., tag 28) are associated with higher engagement.

---

## **2. Data Preprocessing**

All preprocessing steps are implemented in `load_clean.py`:

- Remove duplicates
- Remove `null` / `Nan` / `None` values

---

## **3. Baseline Models**

Implemented in `Baseline_models.ipynb`.

### **A. Popularity-Based Recommender**
- Ranks videos by their total count of positive interactions in the train set.
- Ignores personalization entirely.

**Results:**
- **Precision@10 ≈ 0.75**, **Recall@10 ≈ 0.008**
- Performs well only because the test set contains globally popular items that overlap across users.
- Recall is low for the popularity baseline because it always recommends the same top items to every user, missing many of their actual relevant (but less popular) interactions.
- Lacks personalization and diversity.

---

### **B. Matrix Factorization (SVD - Surprise Library)**
- Uses user/item latent factors to predict binary preferences.
- Trained using 5-fold cross-validation and full-train fit.

**Results:**
- **Precision@10 ≈ 0.02**, **Recall@10 ≈ 0.0001**
- Poor performance due to mismatch with implicit data and sparse signals.

---

## **4. ALS Model**

Implemented in `ALS.ipynb`.

We trained an **Alternating Least Squares (ALS)** model using the `implicit` library:
- Treats binary interactions as confidence-weighted implicit signals.
- Built the item-user confidence matrix, transposed it to get `user_items`.

**Results:**
Due to API/version issues (see https://github.com/benfred/implicit/issues/535 and https://github.com/benfred/implicit/issues/708), I could not manage to compute scores and rank items per user.
But with the theory and praticals seen in class, we can think that ALS will underperform due to:
  - Cold-start issues for unseen items in test.
  - Lack of tuning and user/item side features.
  - Difficulty learning good user embeddings on sparse binary data.

**Conclusion**: ALS needs fine-tuning and fallback mechanisms to handle unseen users/items. In its raw form, it underperforms vs. simpler baselines.

---

## **5. Deep Autoencoder**

Implemented in `AutoEncoder.ipynb` (extended), using PyTorch.

I designed a **2-layer deep autoencoder** that:
- Encodes a user’s interaction vector to a latent space
- Decodes back to reconstruct potential preferences
- Is trained using binary cross-entropy on a dense binary matrix

**Model Configuration**:

    model = DeepAutoEncoder(input_dim=n_items, hidden_dims=[256, 128], dropout=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

### **Why This Model?**
- Non-linear representation power  
- Learns richer user embedding than SVD or ALS  
- More expressive than shallow architectures  
- Handles sparsity via dropout and implicit learning

### **Results:**

=== Deep Autoencoder @K=10 ===

Precision@10: 0.5227

Recall@10   : 0.0915

NDCG@10     : 0.5586

MAP@10      : 0.3996


### **Interpretation:**
- **Precision@10 (0.52)** is significantly better than SVD (~0.02) and even close to popularity (~0.75), but with added personalization.
- **Recall (0.091)** shows the model recovers much more of the user's true interests than popularity (0.008).
- **NDCG and MAP** show that the model not only retrieves positives but also ranks them well.

Compared to all baselines, the Deep Autoencoder offers the best **balance between personalization and ranking quality**.

---

## **6. Final Thoughts and Limitations**

The project reveals a core truth of recommender systems:  
> **How you define “interaction” directly affects what a model learns.**

I chose to define a **positive interaction as `watch_ratio ≥ 2`**, based on the KuaiRec paper’s guidance. This ensures that recommendations are based on **strong signals** (e.g. re-watched videos), not accidental views.

However, this threshold is flexible:
- A **lower threshold** (e.g., `≥ 1`) might capture more positive signals, leading to better **recall** but noisier recommendations.
- A **higher threshold** (e.g., `≥ 3`) would yield stronger confidence but less data.

**All metrics (precision, recall, etc.) are tightly coupled to this modeling choice.**  
Changing it changes both what I optimize and what I measure.

---

## **Conclusion**

My experiments highlight how:
- Simple baselines like popularity are hard to beat in precision, but weak in recall.
- Matrix factorization must be adapted for implicit feedback.
- ALS struggles without rich context.
- A deep autoencoder provides a compelling hybrid between ranking, personalization, and robustness.

> **The "best" model is only as good as the way we define and understand user behavior.**

---

**Authors**:  
- **Léo Lopes**

---
