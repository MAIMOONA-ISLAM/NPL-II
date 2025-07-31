# IMDB Sentiment Analysis

This project performs sentiment analysis on the IMDB 50k movie reviews dataset using machine learning techniques.

## Features

- **Data Loading**: Loads the IMDB dataset from CSV file
- **Text Preprocessing**: Cleans reviews by removing HTML tags, special characters, and stopwords
- **Feature Extraction**: Uses CountVectorizer to convert text to numerical features
- **Model Training**: Trains a Logistic Regression classifier
- **Evaluation**: Provides comprehensive model evaluation metrics
- **Visualizations**: Creates multiple plots including confusion matrix and sentiment distribution

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Make sure the `IMDB Dataset.csv` file is in the same directory as `main.py`
2. Run the script:

```bash
python main.py
```

## Output

The script will:

1. **Load and display dataset information**
2. **Clean the text data** (remove HTML tags, stopwords, etc.)
3. **Split data** into 80% training and 20% testing sets
4. **Train a Logistic Regression model**
5. **Print a detailed classification report**
6. **Generate visualizations**:
   - Confusion Matrix
   - Sentiment Distribution Bar Plot
   - Model Performance Metrics
   - Training vs Testing Set Distribution
7. **Save the plots** as `sentiment_analysis_results.png`
8. **Show sample predictions** with accuracy indicators

## Model Details

- **Algorithm**: Logistic Regression
- **Vectorizer**: CountVectorizer with 5000 max features and bigrams
- **Text Cleaning**: Lowercase conversion, HTML tag removal, stopword removal
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

## Expected Results

The model typically achieves:
- **Accuracy**: ~85-90%
- **Precision**: ~85-90%
- **Recall**: ~85-90%
- **F1-Score**: ~85-90%

## Files

- `main.py`: Main sentiment analysis script
- `requirements.txt`: Python package dependencies
- `IMDB Dataset.csv`: Input dataset (not included in repository)
- `sentiment_analysis_results.png`: Generated visualization (created after running)

## Dataset Format

The CSV file should have two columns:
- `review`: The movie review text
- `sentiment`: Either "positive" or "negative" 