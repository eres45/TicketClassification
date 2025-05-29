import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("NLTK resources downloaded successfully")
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

def clean_text(text):
    """Clean and normalize text"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def preprocess_text(text):
    """Full text preprocessing: cleaning, tokenization, stopword removal, lemmatization"""
    if not isinstance(text, str) or not text:
        return ""
    
    text = clean_text(text)
    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

def main():
    print("Starting simplified ticket classifier...")
    
    try:
        print("Loading data...")
        data_path = 'ai_dev_assignment_tickets_complex_1000.xls'
        data = pd.read_excel(data_path)
        print(f"Loaded {len(data)} records")
        
        print("\nData columns:", data.columns.tolist())
        print("\nMissing values per column:")
        print(data.isnull().sum())
        
        data['ticket_text'] = data['ticket_text'].fillna("")
        
        print("\nPreprocessing a sample of 5 records...")
        for i in range(5):
            original = data.iloc[i]['ticket_text']
            processed = preprocess_text(original)
            print(f"\nOriginal: {original[:100]}...")
            print(f"Processed: {processed[:100]}...")
        
        print("\nSetting up a basic model...")
        
        data['processed_text'] = data['ticket_text'].apply(preprocess_text)
        
        data['issue_type'] = data['issue_type'].fillna('Unknown')
        data['urgency_level'] = data['urgency_level'].fillna('Medium')
        
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(data['processed_text'])
        
        issue_encoder = LabelEncoder()
        urgency_encoder = LabelEncoder()
        
        y_issue = issue_encoder.fit_transform(data['issue_type'])
        y_urgency = urgency_encoder.fit_transform(data['urgency_level'])
        
        print("\nUnique issue types:", data['issue_type'].unique())
        print("Unique urgency levels:", data['urgency_level'].unique())
        
        X_train, X_test, y_train_issue, y_test_issue = train_test_split(
            X, y_issue, test_size=0.2, random_state=42
        )
        
        print("\nTraining a simple Logistic Regression model for issue_type...")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train_issue)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test_issue, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\nSimplified classifier completed successfully")
        
    except Exception as e:
        print(f"\nAn error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
