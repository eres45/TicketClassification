import pandas as pd
import numpy as np
import re
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime

warnings.filterwarnings('ignore')

class SimpleTicketProcessor:
    """A robust ticket processor that doesn't rely on external NLTK downloads"""
    
    def __init__(self, data_path):
        """Initialize the processor with the path to the data file"""
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.encoders = {}
        self.vectorizer = None
        self.product_list = []
        
    def load_data(self):
        """Load data from file and perform initial preprocessing"""
        print("Loading data...")
        self.data = pd.read_excel(self.data_path)
        print(f"Loaded {len(self.data)} records")
        
        self.product_list = self.data['product'].dropna().unique().tolist()
        
        print("\nData columns:", self.data.columns.tolist())
        print("\nMissing values per column:")
        print(self.data.isnull().sum())
        
        self.data['ticket_text'] = self.data['ticket_text'].fillna("")
        self.data['issue_type'] = self.data['issue_type'].fillna('Unknown')
        self.data['urgency_level'] = self.data['urgency_level'].fillna('Medium')
        
        return self
    
    def clean_text(self, text):
        """Clean and normalize text without requiring NLTK"""
        if not isinstance(text, str) or not text:
            return ""
        
        text = text.lower()
        
        text = re.sub(r'[^\w\s]', ' ', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_data(self):
        """Preprocess the entire dataset"""
        print("Preprocessing data...")
        
        self.data['processed_text'] = self.data['ticket_text'].apply(self.clean_text)
        
        self.data['ticket_length'] = self.data['ticket_text'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)
        
        urgent_terms = ['urgent', 'immediately', 'asap', 'emergency', 'critical', 'important']
        self.data['urgency_term_count'] = self.data['ticket_text'].apply(
            lambda x: sum(1 for term in urgent_terms if term in str(x).lower()) if isinstance(x, str) else 0
        )
        
        self.data['word_count'] = self.data['ticket_text'].apply(
            lambda x: len(str(x).split()) if isinstance(x, str) else 0
        )
        
        print("\nSample of preprocessed data:")
        sample_idx = np.random.randint(0, len(self.data), 3)
        for idx in sample_idx:
            print(f"\nOriginal text: {self.data.iloc[idx]['ticket_text'][:100]}...")
            print(f"Processed text: {self.data.iloc[idx]['processed_text'][:100]}...")
            print(f"Features: length={self.data.iloc[idx]['ticket_length']}, "
                  f"word_count={self.data.iloc[idx]['word_count']}, "
                  f"urgency_terms={self.data.iloc[idx]['urgency_term_count']}")
        
        return self
    
    def prepare_features(self):
        """Prepare features for model training"""
        print("\nPreparing features...")
        
        self.vectorizer = TfidfVectorizer(max_features=2000, min_df=2)
        tfidf_features = self.vectorizer.fit_transform(self.data['processed_text'])
        
        tfidf_array = tfidf_features.toarray()
        
        additional_features = self.data[['ticket_length', 'urgency_term_count', 'word_count']].values
        
        X = np.hstack((tfidf_array, additional_features))
        
        target_columns = ['issue_type', 'urgency_level']
        y_encoded = {}
        
        for col in target_columns:
            encoder = LabelEncoder()
            y_encoded[col] = encoder.fit_transform(self.data[col])
            self.encoders[col] = encoder
            
            print(f"\nDistribution of {col}:")
            class_counts = self.data[col].value_counts()
            for cls, count in class_counts.items():
                print(f"  {cls}: {count} ({count/len(self.data)*100:.1f}%)")
        
        return X, y_encoded
    
    def train_models(self, model_types=None):
        """Train classification models"""
        if model_types is None:
            model_types = ['logistic', 'svm', 'rf']
        
        print("\nTraining models...")
        
        X, y_encoded = self.prepare_features()
        
        splits = {}
        for target, y in y_encoded.items():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            splits[target] = {
                'X_train': X_train, 'X_test': X_test, 
                'y_train': y_train, 'y_test': y_test
            }
        
        model_configs = {
            'logistic': LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced'),
            'svm': LinearSVC(max_iter=2000, C=1.0, class_weight='balanced'),
            'rf': RandomForestClassifier(n_estimators=100, class_weight='balanced')
        }
        for target in y_encoded.keys():
            print(f"\nTraining models for {target}...")
            
            best_acc = 0
            best_model = None
            best_type = None
            
            X_train = splits[target]['X_train']
            X_test = splits[target]['X_test']
            y_train = splits[target]['y_train']
            y_test = splits[target]['y_test']
            
            for model_type in model_types:
                if model_type not in model_configs:
                    print(f"Warning: Unsupported model type '{model_type}'. Skipping.")
                    continue
                
                print(f"  Training {model_type}...")
                model = model_configs[model_type]
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"  {model_type} accuracy: {accuracy:.4f}")
                
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_model = model
                    best_type = model_type
            
            if best_model is not None:
                self.models[target] = {'model': best_model, 'type': best_type}
                print(f"  Best model for {target}: {best_type} (accuracy: {best_acc:.4f})")
                
                y_pred = best_model.predict(X_test)
                report = classification_report(y_test, y_pred, target_names=self.encoders[target].classes_)
                print(f"\nClassification report for {target}:")
                print(report)
                
                self.plot_confusion_matrix(
                    y_test, y_pred, 
                    target, 
                    self.encoders[target].classes_
                )
        
        return self
    
    def plot_confusion_matrix(self, y_true, y_pred, title, class_names):
        """Plot a confusion matrix"""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Normalized Confusion Matrix - {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/confusion_matrix_{title.lower().replace(" ", "_")}.png')
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
        
        text_lower = text.lower()
        
        for product in self.product_list:
            if product.lower() in text_lower:
                entities['product'].append(product)
        
        date_patterns = [
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}\b',
            r'\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{2,4}\b',
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{2,4}\b',
            r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b',
            r'\b\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            entities['dates'].extend(matches)
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
        if not self.models or not self.vectorizer:
            raise ValueError("Models must be trained before making predictions")
        
        processed_text = self.clean_text(ticket_text)
        
        tfidf_features = self.vectorizer.transform([processed_text]).toarray()
        
        ticket_length = len(ticket_text) if isinstance(ticket_text, str) else 0
        
        urgent_terms = ['urgent', 'immediately', 'asap', 'emergency', 'critical', 'important']
        urgency_term_count = sum(1 for term in urgent_terms if term in str(ticket_text).lower()) if isinstance(ticket_text, str) else 0
        
        word_count = len(str(ticket_text).split()) if isinstance(ticket_text, str) else 0
        
        additional_features = np.array([[ticket_length, urgency_term_count, word_count]])
        X = np.hstack((tfidf_features, additional_features))
        
        predictions = {}
        for target, model_info in self.models.items():
            model = model_info['model']
            y_pred = model.predict(X)[0]
            predictions[target] = self.encoders[target].inverse_transform([y_pred])[0]
        
        entities = self.extract_entities(ticket_text)
        
        result = {
            'issue_type': predictions.get('issue_type', 'Unknown'),
            'urgency_level': predictions.get('urgency_level', 'Medium'),
            'entities': entities
        }
        
        return result
    
    def batch_predict(self, ticket_texts):
        """Predict issue type, urgency level, and extract entities from multiple tickets"""
        results = []
        for text in ticket_texts:
            results.append(self.predict_ticket_info(text))
        return results

def main():
    print("Starting Ticket Classifier Pipeline...")
    
    try:
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        
        processor = SimpleTicketProcessor('ai_dev_assignment_tickets_complex_1000.xls')
        processor.load_data()
        processor.preprocess_data()
        processor.train_models(['logistic', 'svm', 'rf'])
        
        example_tickets = [
            """
            I purchased a SmartWatch V2 on January 15th, 2023, and it's been nothing but trouble.
            The screen keeps freezing and the battery drains within hours. This is the third time
            I've had issues with this product. I need a replacement as soon as possible, as I
            use this for work daily. My order number is #45678.
            """,
            
            """
            My RoboChef Blender arrived yesterday but is missing the instruction manual. 
            Could you please email me a PDF copy? I'd like to use it this weekend for a family gathering.
            """,
            
            """
            URGENT: My Vision LED TV has completely stopped working after a power surge 
            last night (05/28/2023). I have an important presentation for work tomorrow 
            and need this fixed IMMEDIATELY. This is the second TV I've had issues with 
            from your company. Very disappointed!
            """
        ]
        
        print("\nExample Predictions:")
        for i, ticket in enumerate(example_tickets):
            print(f"\nTicket {i+1}:")
            print(f"Text: {ticket[:100]}...")
            
            prediction = processor.predict_ticket_info(ticket)
            print(f"Predicted Issue Type: {prediction['issue_type']}")
            print(f"Predicted Urgency Level: {prediction['urgency_level']}")
            print("Extracted Entities:")
            for entity_type, entities in prediction['entities'].items():
                print(f"  {entity_type}: {entities}")
        
        print("\nTicket Classifier Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
