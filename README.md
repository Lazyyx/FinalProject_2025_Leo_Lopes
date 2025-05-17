# KuaiRec Recommender System Project

This project was developed as part of the *Recommender Systems* course and centers around the **KuaiRec dataset**, a large-scale, fully-observed dataset from the Kuaishou short-video platform. The goal is to explore, preprocess, and model real-world user-video interactions, and evaluate the effectiveness of several recommendation algorithms.

---

## **Overview**

We implemented multiple stages in this project:
1. **Data Analysis & Visualization**
2. **Data Cleaning and Preprocessing**
3. **Baseline Recommenders**:
   - Popularity-Based
   - Matrix Factorization (SVD)
4. **Alternating Least Squares (ALS)**
5. **Evaluation and Comparison**

The final goal was to understand what works, what doesn't, and *why*, given the structure and dynamics of the dataset.

---

## **1. Data Analysis & Visualization**

We began our exploration in `EDA.ipynb`, a notebook that was already given, which included:
- Distribution of watch ratios
- User/video interaction counts
- Initial sanity checks

We extended this analysis in `EDA_advanced.ipynb` with:
- Statistics on social network, item_categories ..
- Time-series plots of daily activity
- Video duration vs. average watch ratio
- Watch ratio distribution across top-5 video tags

**Key Observations**:
- Most users only have one friend
- Most users watch a small number of videos intensively.
- Shorter videos are more likely to be rewatched (`watch_ratio >= 2`).
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
- **High Precision@K (e.g., ~0.75)** due to popular videos being overrepresented in the test positives.
- **Very Low Recall (~0.008)**: fails to cover the user's broader interests.

**Conclusion**: Works well when a few videos dominate interaction volume, but lacks diversity and personalization.

---

### **B. Matrix Factorization (SVD - Surprise Library)**
- Uses user/item latent factors to predict binary preferences.
- Trained using 5-fold cross-validation and full-train fit.

**Results:**
- **Precision@10 ~0.02**, **Recall@10 ~0.0001**
- **Accuracy ~77%**, but misleading due to strong class imbalance.

**Conclusion**: Classic SVD struggles with implicit, sparse data and doesn't optimize for ranking. Its pointwise loss is poorly suited to top-K tasks.

---

## **4. ALS Model**

Implemented in `ALS.ipynb`.

We trained an **Alternating Least Squares (ALS)** model using the `implicit` library:
- Treats binary interactions as confidence-weighted implicit signals.
- Built the item-user confidence matrix, transposed it to get `user_items`.
Due to API/version issues (see https://github.com/benfred/implicit/issues/535 and https://github.com/benfred/implicit/issues/708), I could not manage to compute scores and rank items per user.

**Results:**
Due to API/version issues (see https://github.com/benfred/implicit/issues/535 and https://github.com/benfred/implicit/issues/708), I could not manage to compute scores and rank items per user.
But with the theory and praticals seen in class, we can think that ALS will underperform due to:
  - Cold-start issues for unseen items in test.
  - Lack of tuning and user/item side features.
  - Difficulty learning good user embeddings on sparse binary data.

**Conclusion**: ALS needs fine-tuning and fallback mechanisms to handle unseen users/items. In its raw form, it underperforms vs. simpler baselines.

---

## **5. Evaluation**

We used standard ranking metrics:
- **Precision@K**
- **Recall@K**
- **NDCG@K**
- **MAP@K**

All models were trained using `big_matrix.py` and evaluated using `small_matrix.csv` as the test set.

---

## **6. Conclusions & Reflections**

- **Popularity-based** works surprisingly well in precision due to skewed interaction distribution.
- **SVD** is not well-suited to binary implicit feedback tasks and fails in recall/ranking.
- **ALS** : API issues made integration challenging. But should underperform anyway.
- Future improvements:
  - Integrate **user and video metadata** (tags, user activity levels).
  - Handle **cold-start** users and items with hybrid approaches.
  - Use ranking-optimized models (e.g., BPR, LightFM) or session-aware deep models.

---

## **Author**
- **Léo Lopes**

---
