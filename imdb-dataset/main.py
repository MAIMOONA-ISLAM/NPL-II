# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data (only first time)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# ✅ Step 1: Load Dataset
print("Loading IMDB Dataset...")
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "IMDB Dataset.csv")

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: Could not find 'IMDB Dataset.csv' in {script_dir}")
    print("Please make sure the CSV file is in the same directory as this script.")
    exit(1)
print(f"Dataset loaded! Shape: {df.shape}")
print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")

# ✅ Step 2: Basic Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove numbers and punctuation
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

# ✅ Step 3: Apply Cleaning
print("\nCleaning reviews...")
df['clean_review'] = df['review'].apply(clean_text)

# ✅ Step 4: Prepare Data
X = df['clean_review']
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)  # 1 = Positive, 0 = Negative

# ✅ Step 5: Train-Test Split (80% training, 20% testing)
print("\nSplitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# ✅ Step 6: Vectorization + Model Pipeline
print("\nTraining Logistic Regression model...")
vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
model = LogisticRegression(random_state=42, max_iter=1000)

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', model)
])

# ✅ Step 7: Train Model
pipeline.fit(X_train, y_train)

# ✅ Step 8: Evaluate
y_pred = pipeline.predict(X_test)
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# ✅ Step 9: Create Visualizations
print("\nCreating visualizations...")

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create a figure with 2x2 subplots for comprehensive analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('IMDB Sentiment Analysis - Complete Results', fontsize=18, fontweight='bold')

# 1. Enhanced Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            ax=axes[0, 0], cbar_kws={'label': 'Count'})
axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Predicted Label', fontsize=12)
axes[0, 0].set_ylabel('Actual Label', fontsize=12)

# Add percentage annotations
total = cm.sum()
for i in range(2):
    for j in range(2):
        percentage = cm[i, j] / total * 100
        axes[0, 0].text(j+0.5, i+0.7, f'{percentage:.1f}%', 
                       ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# 2. Enhanced Sentiment Distribution Bar Plot
sentiment_counts = df['sentiment'].value_counts()
colors = ['#ff6b6b', '#4ecdc4']
bars = axes[0, 1].bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.8)
axes[0, 1].set_title('Dataset Sentiment Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Number of Reviews', fontsize=12)
axes[0, 1].grid(axis='y', alpha=0.3)

# Add value labels on bars with percentage
for bar, count in zip(bars, sentiment_counts.values):
    percentage = count / len(df) * 100
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                   f'{count:,}\n({percentage:.1f}%)', ha='center', va='bottom', 
                   fontweight='bold', fontsize=11)

# 3. Model Performance Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors_metrics = ['#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3']

bars_metrics = axes[1, 0].bar(metrics, values, color=colors_metrics, alpha=0.8)
axes[1, 0].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Score', fontsize=12)
axes[1, 0].set_ylim(0, 1)
axes[1, 0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars_metrics, values):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 4. Training vs Testing Distribution
train_counts = y_train.value_counts()
test_counts = y_test.value_counts()

x = np.arange(2)
width = 0.35

bars_train = axes[1, 1].bar(x - width/2, [train_counts[0], train_counts[1]], width, 
                           label='Training Set', color='#ff9ff3', alpha=0.8)
bars_test = axes[1, 1].bar(x + width/2, [test_counts[0], test_counts[1]], width, 
                          label='Testing Set', color='#54a0ff', alpha=0.8)

axes[1, 1].set_title('Training vs Testing Set Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Number of Reviews', fontsize=12)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(['Negative', 'Positive'])
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars, label in [(bars_train, 'Train'), (bars_test, 'Test')]:
    for bar, count in zip(bars, [train_counts[0], train_counts[1]] if label == 'Train' else [test_counts[0], test_counts[1]]):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                       f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Adjust layout and save
plt.tight_layout()
plt.savefig('sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
print("Visualizations saved as 'sentiment_analysis_results.png'")

# Display the plot
plt.show()

# Create a standalone, larger confusion matrix
print("\nCreating standalone confusion matrix...")
plt.figure(figsize=(10, 8))

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create the heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            cbar_kws={'label': 'Count'})

plt.title('IMDB Sentiment Analysis - Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('Actual Label', fontsize=14)

# Add detailed annotations with percentages
total = cm.sum()
for i in range(2):
    for j in range(2):
        count = cm[i, j]
        percentage = count / total * 100
        plt.text(j+0.5, i+0.7, f'{count}\n({percentage:.1f}%)', 
                ha='center', va='center', fontsize=12, color='white', fontweight='bold')

# Add summary text
plt.figtext(0.5, 0.02, 
           f'Total Predictions: {total:,} | Accuracy: {accuracy:.2%} | '
           f'True Negatives: {cm[0,0]:,} | True Positives: {cm[1,1]:,} | '
           f'False Negatives: {cm[1,0]:,} | False Positives: {cm[0,1]:,}',
           ha='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

plt.tight_layout()
plt.savefig('confusion_matrix_standalone.png', dpi=300, bbox_inches='tight')
print("Standalone confusion matrix saved as 'confusion_matrix_standalone.png'")
plt.show()

# Print summary statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model Precision: {precision:.4f}")
print(f"Model Recall: {recall:.4f}")
print(f"Model F1-Score: {f1:.4f}")

# Show some example predictions
print("\n" + "="*50)
print("SAMPLE PREDICTIONS")
print("="*50)

# Get some random samples
sample_indices = np.random.choice(len(X_test), 5, replace=False)

for i, idx in enumerate(sample_indices):
    review = X_test.iloc[idx]
    true_label = y_test.iloc[idx]
    pred_label = y_pred[idx]
    
    print(f"\nSample {i+1}:")
    print(f"Review (first 100 chars): {review[:100]}...")
    print(f"True Sentiment: {'Positive' if true_label == 1 else 'Negative'}")
    print(f"Predicted Sentiment: {'Positive' if pred_label == 1 else 'Negative'}")
    print(f"Correct: {'✓' if true_label == pred_label else '✗'}")

print("\nScript completed successfully!")

