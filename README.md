---
TikTok Video Claim Classification Project
---
----


Project Overview
----

This project aims to classify TikTok videos into Claim or Opinion categories based on video metadata, author info, and video transcription text. The goal is to develop and compare multiple machine learning models to accurately predict the claim status of TikTok videos.


Dataset Summary
----
Total records: 19,382 videos initially.

Key features:

video_duration_sec, video_view_count, video_like_count, video_share_count, video_comment_count, video_download_count

Author attributes: author_ban_status, verified_status

Textual data: video_transcription_text

Target: claim_status (Claim vs Opinion)

Data Cleaning and Preprocessing
----
Removed rows with missing critical data (reducing dataset to ~19,000 rows).

Handled outliers in numerical columns using the IQR method (e.g., text_length).

Encoded target variable claim_status as binary: claim = 1, opinion = 0.

One-hot encoded categorical variables (author_ban_status, verified_status).

Created normalized engagement metrics:

likes_per_view

shares_per_view

comments_per_view

Processed video transcription text using TF-IDF vectorization followed by Truncated SVD for dimensionality reduction.

Exploratory Data Analysis (EDA)
----
Significant statistical difference found in text_length between claim and opinion videos (p-value << 0.05).

Engagement metrics (likes_per_view, shares_per_view, comments_per_view) were analyzed across claim/opinion and user ban status.

Visualizations (boxplots, countplots) provided insight into feature distributions and class balances.

[Insert EDA Visualizations Here]
Examples: Boxplots of likes_per_view by claim_status, bar plots of verified_status counts, etc.

Modeling Approaches
----
Three classification models were trained and evaluated:

Model	Key Details
Logistic Regression	Baseline linear model
Random Forest	Ensemble tree-based model
XGBoost	Gradient boosting tree algorithm

Models were tuned via GridSearchCV with cross-validation.

Used all numeric, categorical, and reduced textual features.

Results Summary
----
Metric	Logistic Regression	Random Forest	XGBoost
Accuracy	~ (not specified)	99.77%	99.87%
Precision	(not specified)	1.00	1.00
Recall	(not specified)	0.9934	0.9963
F1-Score	(not specified)	0.9967	0.9981

XGBoost outperformed both Logistic Regression and Random Forest across all key metrics.

The model shows excellent precision and recall, indicating balanced performance for both classes.

Confusion Matrices
[Insert Confusion Matrix Images for each model here]

Random Forest confusion matrix shows very high true positive and true negative rates.

XGBoost confusion matrix further improves recall on the positive class (claim).


Feature Importance and Interpretation
----
Key features impacting classification include:

Text length

Engagement ratios (likes, shares, comments per view)

Author verification and ban status

TF-IDF components from video transcription text also contributed meaningfully to model performance.

[Insert Feature Importance Plot Here]


Conclusion
----
The classification models demonstrate strong ability to distinguish between claim and opinion TikTok videos.

XGBoost proved to be the most effective model with the highest accuracy and balanced precision/recall.

Integrating textual features with metadata and author attributes significantly improves classification.

The methodology can be extended to other social media platforms and content moderation tasks.

