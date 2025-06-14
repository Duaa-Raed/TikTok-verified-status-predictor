
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
tiktok_dataset = pd.read_csv("//tiktok_dataset.csv")

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
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=tiktok_dataset_cleaned, x='author_ban_status', y='likes_per_view', palette='inferno')
plt.title('Boxplot: likes_per_view by author_ban_status')
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
plt.show()

# Feature coefficients
feature_coefficients = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_model.coef_[0]
}).sort_values('coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_coefficients, x='coefficient', y='feature', palette='mako')
plt.title('Feature Coefficients')
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
plt.show()









