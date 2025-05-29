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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import dateparser
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Download NLTK resources (uncomment these on first run)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TicketProcessor:
    def __init__(self, data_path):
        """Initialize the processor with the path to the data file"""
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train_issue = None
        self.y_test_issue = None
        self.y_train_urgency = None
        self.y_test_urgency = None
        self.issue_type_model = None
        self.urgency_level_model = None
        self.issue_encoder = None
        self.urgency_encoder = None
        self.vectorizer = None
        self.product_list = None
        
    def load_data(self):
        """Load data from file and perform initial preprocessing"""
        print("Loading data...")
        self.data = pd.read_excel(self.data_path)
        print(f"Loaded {len(self.data)} records")
        
        # Create a list of products for entity extraction
        self.product_list = self.data['product'].dropna().unique().tolist()
        
        # Handle missing values in ticket_text
        self.data['ticket_text'] = self.data['ticket_text'].fillna("")
        
        return self
    
    def clean_text(self, text):
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
    
    def preprocess_text(self, text):
        """Full text preprocessing: cleaning, tokenization, stopword removal, lemmatization"""
        if not isinstance(text, str) or not text:
            return ""
        
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Join tokens back into text
        return " ".join(tokens)
    
    def preprocess_data(self):
        """Preprocess the entire dataset"""
        print("Preprocessing data...")
        
        # Apply preprocessing to ticket text
        self.data['processed_text'] = self.data['ticket_text'].apply(self.preprocess_text)
        
        # Feature: ticket length
        self.data['ticket_length'] = self.data['ticket_text'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)
        
        # Feature: sentiment score
        self.data['sentiment'] = self.data['ticket_text'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if isinstance(x, str) else 0
        )
        
        # Handle missing values in target variables
        self.data['issue_type'] = self.data['issue_type'].fillna('Unknown')
        self.data['urgency_level'] = self.data['urgency_level'].fillna('Medium')
        
        return self
    
    def extract_features(self):
        """Extract features for model training"""
        print("Extracting features...")
        
        # Create and fit the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000)
        text_features = self.vectorizer.fit_transform(self.data['processed_text'])
        
        # Convert to DataFrame for easier manipulation
        text_features_df = pd.DataFrame(text_features.toarray(), 
                                        columns=self.vectorizer.get_feature_names_out())
        
        # Add additional features
        features_df = pd.concat([
            text_features_df,
            self.data[['ticket_length', 'sentiment']]
        ], axis=1)
        
        return features_df
    
    def split_data(self):
        """Split data into train and test sets"""
        print("Splitting data...")
        
        # Extract features
        features = self.extract_features()
        
        # Encode target variables
        self.issue_encoder = LabelEncoder()
        self.urgency_encoder = LabelEncoder()
        
        issue_encoded = self.issue_encoder.fit_transform(self.data['issue_type'])
        urgency_encoded = self.urgency_encoder.fit_transform(self.data['urgency_level'])
        
        # Split data
        X_train, X_test, y_train_issue, y_test_issue = train_test_split(
            features, issue_encoded, test_size=0.2, random_state=42
        )
        
        _, _, y_train_urgency, y_test_urgency = train_test_split(
            features, urgency_encoded, test_size=0.2, random_state=42
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train_issue = y_train_issue
        self.y_test_issue = y_test_issue
        self.y_train_urgency = y_train_urgency
        self.y_test_urgency = y_test_urgency
        
        return self
    
    def train_models(self, model_type='logistic'):
        """Train classification models"""
        print(f"Training models using {model_type}...")
        
        if model_type == 'logistic':
            issue_model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='auto')
            urgency_model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='auto')
        elif model_type == 'svm':
            issue_model = SVC(probability=True)
            urgency_model = SVC(probability=True)
        elif model_type == 'rf':
            issue_model = RandomForestClassifier(n_estimators=100)
            urgency_model = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError("Unsupported model type. Choose 'logistic', 'svm', or 'rf'")
        
        # Train issue type model
        issue_model.fit(self.X_train, self.y_train_issue)
        
        # Train urgency level model
        urgency_model.fit(self.X_train, self.y_train_urgency)
        
        self.issue_type_model = issue_model
        self.urgency_level_model = urgency_model
        
        return self
    
    def evaluate_models(self):
        """Evaluate trained models"""
        print("Evaluating models...")
        
        # Evaluate issue type model
        issue_pred = self.issue_type_model.predict(self.X_test)
        issue_accuracy = accuracy_score(self.y_test_issue, issue_pred)
        issue_report = classification_report(self.y_test_issue, issue_pred)
        
        # Evaluate urgency level model
        urgency_pred = self.urgency_level_model.predict(self.X_test)
        urgency_accuracy = accuracy_score(self.y_test_urgency, urgency_pred)
        urgency_report = classification_report(self.y_test_urgency, urgency_pred)
        
        print(f"Issue Type Model Accuracy: {issue_accuracy:.4f}")
        print("Issue Type Classification Report:")
        print(issue_report)
        
        print(f"Urgency Level Model Accuracy: {urgency_accuracy:.4f}")
        print("Urgency Level Classification Report:")
        print(urgency_report)
        
        # Plot confusion matrices
        self.plot_confusion_matrix(self.y_test_issue, issue_pred, 'Issue Type')
        self.plot_confusion_matrix(self.y_test_urgency, urgency_pred, 'Urgency Level')
        
        return self
    
    def plot_confusion_matrix(self, y_true, y_pred, title):
        """Plot a confusion matrix"""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{title.lower().replace(" ", "_")}.png')
        plt.close()
    
    def extract_entities(self, text):
        """Extract entities from ticket text"""
        entities = {
            'product': [],
            'dates': [],
            'complaint_keywords': []
        }
        
        if not isinstance(text, str) or not text:
            return entities
        
        # Clean text for processing but keep original for pattern matching
        text_lower = text.lower()
        
        # Extract product names
        for product in self.product_list:
            if product.lower() in text_lower:
                entities['product'].append(product)
        
        # Extract dates using dateparser
        # First, find potential date patterns
        date_patterns = [
            r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}',  # DD/MM/YYYY or similar
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',  # DD Mon YYYY
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}'  # Mon DD, YYYY
        ]
        
        date_matches = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            date_matches.extend(matches)
        
        # Parse potential dates
        for date_str in date_matches:
            parsed_date = dateparser.parse(date_str)
            if parsed_date:
                entities['dates'].append(parsed_date.strftime('%Y-%m-%d'))
        
        # Extract complaint keywords
        complaint_keywords = [
            'broken', 'error', 'issue', 'problem', 'fail', 'not working', 'defect',
            'malfunction', 'damaged', 'late', 'delay', 'slow', 'poor', 'bad',
            'wrong', 'incorrect', 'missing', 'lost', 'stuck', 'frozen', 'crash',
            'bug', 'glitch', 'faulty', 'defective', 'dissatisfied', 'disappointed',
            'unhappy', 'refund', 'return', 'replace', 'repair', 'fix'
        ]
        
        for keyword in complaint_keywords:
            if keyword in text_lower:
                entities['complaint_keywords'].append(keyword)
        
        return entities
    
    def predict_ticket_info(self, ticket_text):
        """Predict issue type, urgency level, and extract entities from a new ticket"""
        if not self.issue_type_model or not self.urgency_level_model:
            raise ValueError("Models must be trained before making predictions")
        
        # Preprocess the text
        processed_text = self.preprocess_text(ticket_text)
        
        # Calculate additional features
        ticket_length = len(ticket_text) if isinstance(ticket_text, str) else 0
        sentiment = TextBlob(str(ticket_text)).sentiment.polarity
        
        # Vectorize the text
        text_features = self.vectorizer.transform([processed_text]).toarray()
        
        # Create a DataFrame with all features
        features_df = pd.DataFrame(text_features, columns=self.vectorizer.get_feature_names_out())
        features_df['ticket_length'] = ticket_length
        features_df['sentiment'] = sentiment
        
        # Predict issue type and urgency level
        issue_type_encoded = self.issue_type_model.predict(features_df)[0]
        urgency_level_encoded = self.urgency_level_model.predict(features_df)[0]
        
        # Decode predictions
        issue_type = self.issue_encoder.inverse_transform([issue_type_encoded])[0]
        urgency_level = self.urgency_encoder.inverse_transform([urgency_level_encoded])[0]
        
        # Extract entities
        entities = self.extract_entities(ticket_text)
        
        # Prepare result
        result = {
            'issue_type': issue_type,
            'urgency_level': urgency_level,
            'entities': entities
        }
        
        return result

# Example usage
def main():
    # Initialize and train the model
    processor = TicketProcessor('ai_dev_assignment_tickets_complex_1000.xls')
    processor.load_data()
    processor.preprocess_data()
    processor.split_data()
    processor.train_models(model_type='logistic')  # 'logistic', 'svm', or 'rf'
    processor.evaluate_models()
    
    # Example prediction
    example_ticket = """
    I purchased a SmartWatch V2 on January 15th, 2023, and it's been nothing but trouble.
    The screen keeps freezing and the battery drains within hours. This is the third time
    I've had issues with this product. I need a replacement as soon as possible, as I
    use this for work daily. My order number is #45678.
    """
    
    prediction = processor.predict_ticket_info(example_ticket)
    print("\nExample Prediction:")
    print(f"Issue Type: {prediction['issue_type']}")
    print(f"Urgency Level: {prediction['urgency_level']}")
    print("Entities:")
    for entity_type, entities in prediction['entities'].items():
        print(f"  {entity_type}: {entities}")

if __name__ == "__main__":
    main()
