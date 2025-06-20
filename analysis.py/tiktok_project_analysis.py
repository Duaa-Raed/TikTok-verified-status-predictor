
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm



# Read the data
tiktok_dataset = pd.read_csv("C:/Users/duaar/OneDrive/Desktop/Tik Tok/tiktok_dataset.csv")

print("Overview of first 10 rows")
print(tiktok_dataset.head(10))
print("\n=== Data Shape (Rows and Columns) ===")
print(tiktok_dataset.shape)
print("\n=== Column Names ===")
print(tiktok_dataset.columns.tolist())
print("\n=== Detailed Column Information ===")
print(tiktok_dataset.info())
print("\n=== Statistical Description of Numeric Data ===")
print(tiktok_dataset.describe())
print("\n=== Number of Missing Values per Column ===")
print(tiktok_dataset.isnull().sum())
# Remove rows with missing values
tiktok_dataset_cleaned = tiktok_dataset.dropna()
print("\n=== Number of Rows After Removing Missing Values ===")
print(tiktok_dataset_cleaned.shape)
# Define function to handle outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Handle outliers in numeric columns
numeric_columns = ['video_duration_sec', 'video_view_count', 'video_like_count',
                   'video_share_count', 'video_download_count', 'video_comment_count']

for column in numeric_columns:
    before_rows = tiktok_dataset_cleaned.shape[0]
    df_cleaned = remove_outliers(tiktok_dataset_cleaned, column)
    after_rows = tiktok_dataset_cleaned.shape[0]
    print(f"\n=== Number of Rows Before Outlier Treatment in {column}: {before_rows}")
    print(f"=== Number of Rows After Outlier Treatment in {column}: {after_rows}")

# Analyze unique values in text columns
    text_columns = ['claim_status', 'verified_status', 'author_ban_status']

    for column in text_columns:
        print(f"\n=== Unique Values in {column} ===")
        print(tiktok_dataset_cleaned[column].value_counts())

# Plot claim_status distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=tiktok_dataset_cleaned, x='claim_status', palette='viridis')
plt.title('Distribution of claim_status')
plt.savefig("C:/Users/duaar/OneDrive/Desktop/images.1/Distribution_of_claim_status.png")
plt.show()

# Code a correlation matrix to help determine most correlated variables
numeric_data = tiktok_dataset_cleaned[numeric_columns]
correlation_matrix = numeric_data.corr()
print("\n=== Correlation Matrix ===")
print(correlation_matrix)

# Create a heatmap to visualize how correlated variables are
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='magma', center=0)
plt.title('Correlation Matrix Heatmap')
plt.savefig("C:/Users/duaar/OneDrive/Desktop/images.1/Correlation_Matrix_Heatmap.png")
plt.show()

# Fix warnings using .loc
tiktok_dataset_cleaned.loc[:, 'likes_per_view'] = tiktok_dataset_cleaned['video_like_count'] / tiktok_dataset_cleaned['video_view_count']
tiktok_dataset_cleaned.loc[:, 'shares_per_view'] = tiktok_dataset_cleaned['video_share_count'] / tiktok_dataset_cleaned['video_view_count']
tiktok_dataset_cleaned.loc[:, 'comments_per_view'] = tiktok_dataset_cleaned['video_comment_count'] / tiktok_dataset_cleaned['video_view_count']

# Remove any illogical or infinite values resulting from division
tiktok_dataset_cleaned = tiktok_dataset_cleaned.replace([float("inf"), -float("inf")], pd.NA).dropna(subset=['likes_per_view', 'shares_per_view', 'comments_per_view'])

# === Analysis Hypotheses ===
print("\n=== Statistical Hypotheses ===")
print("""
First Hypothesis:
H0: No significant difference in likes_per_view between claim and opinion
H1: There is a significant difference in likes_per_view between claim and opinion

Second Hypothesis:
H0: No significant difference in shares_per_view between claim and opinion
H1: There is a significant difference in shares_per_view between claim and opinion

Third Hypothesis:
H0: No significant difference in comments_per_view between claim and opinion
H1: There is a significant difference in comments_per_view between claim and opinion

Fourth Hypothesis:
H0: No significant difference in likes_per_view between banned and active users
H1: There is a significant difference in likes_per_view between banned and active users
""")

# === Box Plots ===
plt.figure(figsize=(10, 6))
sns.boxplot(data=tiktok_dataset_cleaned, x='claim_status', y='likes_per_view', palette='plasma')
plt.title('Boxplot: likes_per_view by claim_status')
plt.savefig("C:/Users/duaar/OneDrive/Desktop/images.1/Boxplot_likes_per_view_by_claim_status.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=tiktok_dataset_cleaned, x='author_ban_status', y='likes_per_view', palette='inferno')
plt.title('Boxplot: likes_per_view by author_ban_status')
plt.savefig("C:/Users/duaar/OneDrive/Desktop/images.1/Boxplot_likes_per_view_by_author_ban_status.png")
plt.show()


# Check target class balance
print("\n=== Target Class Balance (verified_status) ===")
print(tiktok_dataset_cleaned['verified_status'].value_counts())

# Handle class imbalance through upsampling
from sklearn.utils import resample

# Split data into majority and minority classes
majority_class = tiktok_dataset_cleaned[tiktok_dataset_cleaned["verified_status"] == "not verified"]
minority_class = tiktok_dataset_cleaned[tiktok_dataset_cleaned["verified_status"] == "verified"]

# Resample minority class
minority_upsampled = resample(minority_class,
                              replace=True,
                              n_samples=len(majority_class),
                              random_state=0)

# Combine data
balanced_data = pd.concat([majority_class, minority_upsampled]).reset_index(drop=True)

# Check new balance
print("\n=== Target Class Balance After Treatment ===")
print(balanced_data['verified_status'].value_counts())

# Plot showing new balance
plt.figure(figsize=(8, 6))
sns.countplot(data=balanced_data, x='verified_status', palette='rocket')
plt.title('Distribution of verified_status After Balancing')
plt.savefig("C:/Users/duaar/OneDrive/Desktop/images.1/Distribution_of_verified_status_After_Balancing.png")
plt.show()


balanced_data['text_length'] = balanced_data['video_transcription_text'].apply(lambda x: len(str(x)))

# Convert categorical variables to numeric
X = balanced_data[['claim_status', 'author_ban_status', 'text_length']]
y = balanced_data['verified_status']

# Convert categorical variables to numeric
X = pd.get_dummies(X, columns=['claim_status', 'author_ban_status'])

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

# Evaluate model
from sklearn.metrics import classification_report, confusion_matrix
y_pred = lr_model.predict(X_test)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate accuracy manually from confusion matrix
true_positives = conf_matrix[1][1]  # TP
true_negatives = conf_matrix[0][0]  # TN
total_samples = conf_matrix.sum()   # Total elements

manual_accuracy = (true_positives + true_negatives) / total_samples

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(conf_matrix)

print("\n=== Manually Calculated Accuracy ===")
print(f"Accuracy = (TP + TN) / Total = ({true_positives} + {true_negatives}) / {total_samples} = {manual_accuracy:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("C:/Users/duaar/OneDrive/Desktop/images.1/Confusion_Matrix.png")
plt.show()

# Feature coefficients
feature_coefficients = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_model.coef_[0]
}).sort_values('coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_coefficients, x='coefficient', y='feature', palette='mako')
plt.title('Feature Coefficients')
plt.savefig("C:/Users/duaar/OneDrive/Desktop/images.1/Feature_Coefficients.png")
plt.show()


# Compute values for confusion matrix
tn, fp, fn, tp = conf_matrix.ravel()

# Create display of confusion matrix
conf_matrix_display = pd.DataFrame({
    'Predicted Negative': [tn, fn],
    'Predicted Positive': [fp, tp]
}, index=['Actual Negative', 'Actual Positive'])

print("\n=== Detailed Confusion Matrix ===")
print(conf_matrix_display)

# Plot confusion matrix with custom labels
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_display, annot=True, fmt='d', cmap='PuBu',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix with Detailed Labels')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("C:/Users/duaar/OneDrive/Desktop/images.1/Confusion_Matrix_with_Detailed_Labels.png")
plt.show()

# Histogram of text_length
plt.figure(figsize=(10, 6))
sns.histplot(X['text_length'], bins=30, kde=True)
plt.title('Distribution of Text Length (رسم هيستوغرام لتوزيع طول النص)')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

# Statistical analysis of text length differences between "Claim" and "Opinion"
# This analysis uses the 'balanced_data' DataFrame, which should be available in the script's scope.
print("\n=== Statistical Analysis of Text Length: Claim vs Opinion (using balanced_data) ===")

# Separate the text length data for each class from the balanced dataset
claim_texts_length = balanced_data[balanced_data['claim_status'] == 'claim']['text_length']
opinion_texts_length = balanced_data[balanced_data['claim_status'] == 'opinion']['text_length']

# Descriptive statistics for each group
print("\nDescriptive Statistics for Text Length (Claim):")
print(claim_texts_length.describe())
print("\nDescriptive Statistics for Text Length (Opinion):")
print(opinion_texts_length.describe())

# Visualization: Box plot to compare distributions
plt.figure(figsize=(12, 7))
sns.boxplot(x='claim_status', y='text_length', data=balanced_data, palette='coolwarm', order=['opinion', 'claim'])
plt.title('Comparison of Text Length Distribution between Claim and Opinion')
plt.xlabel('Status')
plt.ylabel('Text Length')
plt.show()

# Perform independent t-test to check for statistical significance
# Note: Ensure 'from scipy.stats import ttest_ind' is at the top of your script.
t_stat, p_value = ttest_ind(claim_texts_length, opinion_texts_length)

print("\n--- Independent T-Test Results for text_length between claim and opinion ---")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret the result
alpha = 0.05
if p_value < alpha:
    print("\nConclusion: The difference in mean text length between 'Claim' and 'Opinion' is statistically significant.")
else:
    print("\nConclusion: The difference in mean text length between 'Claim' and 'Opinion' is not statistically significant.")


print("------------------------------------------------------")
# 2. التعامل مع القيم المفقودة، التكرارات، والقيم الشاذة
# تم التعامل مع القيم المفقودة والتكرارات مسبقًا.
# الآن، سنتعامل مع القيم الشاذة في 'text_length' ونقوم بترميز المتغيرات التصنيفية.

# إزالة القيم الشاذة من 'text_length' باستخدام طريقة المدى الربيعي (IQR)
print("\n=== Handling Outliers in text_length using IQR method ===")
print(f"Original shape of balanced_data: {balanced_data.shape}")

Q1 = balanced_data['text_length'].quantile(0.25)
Q3 = balanced_data['text_length'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Identifying outliers in 'text_length' outside the range ({lower_bound:.2f}, {upper_bound:.2f})")

# تحديد عدد القيم الشاذة
outliers_count = balanced_data[(balanced_data['text_length'] < lower_bound) | (balanced_data['text_length'] > upper_bound)].shape[0]
print(f"Number of outliers found: {outliers_count}")

# إزالة القيم الشاذة
balanced_data = balanced_data[(balanced_data['text_length'] >= lower_bound) & (balanced_data['text_length'] <= upper_bound)]
print(f"Shape of data after removing outliers: {balanced_data.shape}")
print("Outliers have been successfully removed from 'balanced_data'.")

print("\n------------------------------------------------------")

# ترميز متغير الهدف (claim_status) والمتغيرات التصنيفية
print("\n=== Encoding Target Variable (claim_status) ===")
print("Original 'claim_status' values:")
print(balanced_data['claim_status'].value_counts())

# ترميز 'claim_status' إلى قيم رقمية: claim=1, opinion=0
balanced_data['claim_status'] = balanced_data['claim_status'].map({'claim': 1, 'opinion': 0})

print("\n'claim_status' values after encoding:")
print(balanced_data['claim_status'].value_counts())
print("\nData head after encoding:")
print(balanced_data.head())

print("------------------------------------------------------")
# 3. Data Preparation for Modeling
print("\n=== Step 3: Feature Engineering and Model Preparation ===")

# حذف حساب ميزات لكل متابع لأنه عمود author_follower_count غير موجود
# balanced_data['likes_per_follower'] = balanced_data['video_like_count'] / balanced_data['author_follower_count']
# balanced_data['shares_per_follower'] = balanced_data['video_share_count'] / balanced_data['author_follower_count']
# balanced_data['comments_per_follower'] = balanced_data['video_comment_count'] / balanced_data['author_follower_count']

# نحدد فقط الميزات الرقمية والتصنيفية المختارة (بدون النص الأصلي)
selected_features = [
    'author_ban_status', 
    'verified_status', 
    'text_length', 
    'likes_per_view', 
    'shares_per_view', 
    'comments_per_view'
    # تم حذف الأعمدة التي تعتمد على author_follower_count
]

# تأكد من وجود الأعمدة المطلوبة
missing_cols = [col for col in selected_features if col not in balanced_data.columns]
if missing_cols:
    raise ValueError(f"Missing columns in balanced_data: {missing_cols}")

# تعريف الهدف والميزات
y = balanced_data['claim_status']
X_basic = balanced_data[selected_features].copy()

# ترميز المتغيرات التصنيفية فقط (بدون النص)
categorical_cols = ['author_ban_status', 'verified_status']
print(f"Encoding categorical columns: {categorical_cols}")
X_basic = pd.get_dummies(X_basic, columns=categorical_cols, drop_first=True)

# معالجة النص باستخدام TF-IDF + تقليل أبعاد (اختياري)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

print("Applying TF-IDF and dimensionality reduction on text column...")
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(balanced_data['video_transcription_text'])  # تأكد من اسم العمود الصحيح للنص

# تقليل الأبعاد (مثلاً إلى 30 بعد فقط)
svd = TruncatedSVD(n_components=30, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

# تحويل النتائج إلى DataFrame مع أسماء أعمدة مناسبة
tfidf_df = pd.DataFrame(tfidf_reduced, columns=[f'tfidf_{i+1}' for i in range(tfidf_reduced.shape[1])], index=balanced_data.index)

# دمج الميزات الرقمية/التصنيفية مع ميزات النص المختصرة
X = pd.concat([X_basic, tfidf_df], axis=1)

print("\nHead of the final features (X):")
print(X.head())

print("\n------------------------------------------------------")

# أكمل تقسيم البيانات وبناء النماذج بعد ذلك كما في كودك السابق
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------- Random Forest -----------------
print("\n--- Random Forest ---")
rf = RandomForestClassifier(random_state=42)

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(
    rf, rf_param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
)
rf_grid.fit(X_train, y_train)

print(f"Best RF Params: {rf_grid.best_params_}")

rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, digits=4))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1-Score:", f1_score(y_test, y_pred_rf))

# ----------------- XGBoost -----------------
print("\n--- XGBoost ---")
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.3]
}

xgb_grid = GridSearchCV(
    xgb, xgb_param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
)
xgb_grid.fit(X_train, y_train)

print(f"Best XGB Params: {xgb_grid.best_params_}")

xgb_best = xgb_grid.best_estimator_
y_pred_xgb = xgb_best.predict(X_test)

print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb, digits=4))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("F1-Score:", f1_score(y_test, y_pred_xgb))












