import gradio as gr
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

class TicketProcessor:
    """A robust ticket processor that classifies and extracts entities from support tickets"""
    
    def __init__(self, data_path=None):
        """Initialize the processor with the path to the data file"""
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.encoders = {}
        self.vectorizer = None
        self.product_list = []
        
    def load_data(self):
        """Load data from file and perform initial preprocessing"""
        if self.data_path is None:
            raise ValueError("Data path must be provided")
            
        self.data = pd.read_excel(self.data_path)
        
        self.product_list = self.data['product'].dropna().unique().tolist()
        
        self.data['ticket_text'] = self.data['ticket_text'].fillna("")
        self.data['issue_type'] = self.data['issue_type'].fillna('Unknown')
        self.data['urgency_level'] = self.data['urgency_level'].fillna('Medium')
        
        return self
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not isinstance(text, str) or not text:
            return ""
        
        text = text.lower()
        
        text = re.sub(r'[^\w\s]', ' ', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_data(self):
        """Preprocess the entire dataset"""
        self.data['processed_text'] = self.data['ticket_text'].apply(self.clean_text)
        
        self.data['ticket_length'] = self.data['ticket_text'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)
        
        urgent_terms = ['urgent', 'immediately', 'asap', 'emergency', 'critical', 'important']
        self.data['urgency_term_count'] = self.data['ticket_text'].apply(
            lambda x: sum(1 for term in urgent_terms if term in str(x).lower()) if isinstance(x, str) else 0
        )
        
        self.data['word_count'] = self.data['ticket_text'].apply(
            lambda x: len(str(x).split()) if isinstance(x, str) else 0
        )
        
        return self
    
    def prepare_features(self):
        """Prepare features for model training"""
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(max_features=2000, min_df=2)
        tfidf_features = self.vectorizer.fit_transform(self.data['processed_text'])
        
        # Convert to dense array for easier manipulation
        tfidf_array = tfidf_features.toarray()
        
        # Combine with other features
        additional_features = self.data[['ticket_length', 'urgency_term_count', 'word_count']].values
        
        # Combine all features
        X = np.hstack((tfidf_array, additional_features))
        
        # Encode target variables
        target_columns = ['issue_type', 'urgency_level']
        y_encoded = {}
        
        for col in target_columns:
            encoder = LabelEncoder()
            y_encoded[col] = encoder.fit_transform(self.data[col])
            self.encoders[col] = encoder
        
        return X, y_encoded
    
    def train_models(self, model_types=None):
        """Train classification models"""
        if model_types is None:
            model_types = ['logistic', 'svm', 'rf']
        
        # Prepare features and targets
        X, y_encoded = self.prepare_features()
        
        # Split data
        splits = {}
        for target, y in y_encoded.items():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            splits[target] = {
                'X_train': X_train, 'X_test': X_test, 
                'y_train': y_train, 'y_test': y_test
            }
        
        # Define model configurations
        model_configs = {
            'logistic': LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced'),
            'svm': LinearSVC(max_iter=2000, C=1.0, class_weight='balanced'),
            'rf': RandomForestClassifier(n_estimators=100, class_weight='balanced')
        }
        
        # Train and evaluate models
        for target in y_encoded.keys():
            best_acc = 0
            best_model = None
            best_type = None
            
            X_train = splits[target]['X_train']
            X_test = splits[target]['X_test']
            y_train = splits[target]['y_train']
            y_test = splits[target]['y_test']
            
            for model_type in model_types:
                if model_type not in model_configs:
                    continue
                
                model = model_configs[model_type]
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Save the best model
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_model = model
                    best_type = model_type
            
            # Save the best model
            if best_model is not None:
                self.models[target] = {'model': best_model, 'type': best_type}
        
        return self
    
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
        
        # Extract dates using regex patterns
        date_patterns = [
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}\b',  # Jan 1, 2023
            r'\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{2,4}\b',  # 1 Jan 2023
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{2,4}\b',  # January 1, 2023
            r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b',  # 01/01/2023
            r'\b\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\b'   # 2023/01/01
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            entities['dates'].extend(matches)
        
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
        if not self.models or not self.vectorizer:
            raise ValueError("Models must be trained before making predictions")
        
        # Preprocess the text
        processed_text = self.clean_text(ticket_text)
        
        # Extract features
        tfidf_features = self.vectorizer.transform([processed_text]).toarray()
        
        # Calculate additional features
        ticket_length = len(ticket_text) if isinstance(ticket_text, str) else 0
        
        # Count urgency terms
        urgent_terms = ['urgent', 'immediately', 'asap', 'emergency', 'critical', 'important']
        urgency_term_count = sum(1 for term in urgent_terms if term in str(ticket_text).lower()) if isinstance(ticket_text, str) else 0
        
        # Count words
        word_count = len(str(ticket_text).split()) if isinstance(ticket_text, str) else 0
        
        # Combine features
        additional_features = np.array([[ticket_length, urgency_term_count, word_count]])
        X = np.hstack((tfidf_features, additional_features))
        
        # Make predictions
        predictions = {}
        for target, model_info in self.models.items():
            model = model_info['model']
            y_pred = model.predict(X)[0]
            predictions[target] = self.encoders[target].inverse_transform([y_pred])[0]
        
        # Extract entities
        entities = self.extract_entities(ticket_text)
        
        # Prepare result
        result = {
            'issue_type': predictions.get('issue_type', 'Unknown'),
            'urgency_level': predictions.get('urgency_level', 'Medium'),
            'entities': entities
        }
        
        return result

processor = None

def load_and_train_model():
    global processor
    status_html = "<div style='color: blue; font-weight: bold;'>Loading data and training models...</div>"
    yield status_html
    
    try:
        processor = TicketProcessor('ai_dev_assignment_tickets_complex_1000.xls')
        processor.load_data()
        processor.preprocess_data()
        processor.train_models(['logistic', 'svm', 'rf'])
        
        status_html = "<div style='color: green; font-weight: bold;'>✓ Models trained successfully!</div>"
    except Exception as e:
        status_html = f"<div style='color: red; font-weight: bold;'>Error: {str(e)}</div>"
    
    yield status_html

def predict(ticket_text):
    global processor
    
    if processor is None:
        return "Please train the model first by clicking 'Load and Train Model'"
    
    if not ticket_text or ticket_text.strip() == "":
        return "Please enter a ticket text"
    
    try:
        result = processor.predict_ticket_info(ticket_text)
        
        output = f"<h3>Prediction Results</h3>"
        output += f"<p><b>Issue Type:</b> {result['issue_type']}</p>"
        output += f"<p><b>Urgency Level:</b> {result['urgency_level']}</p>"
        
        output += "<h4>Extracted Entities:</h4>"
        output += "<ul>"
        
        if result['entities']['product']:
            output += f"<li><b>Products:</b> {', '.join(result['entities']['product'])}</li>"
        else:
            output += "<li><b>Products:</b> None detected</li>"
        
        if result['entities']['dates']:
            output += f"<li><b>Dates:</b> {', '.join(result['entities']['dates'])}</li>"
        else:
            output += "<li><b>Dates:</b> None detected</li>"
        
        if result['entities']['complaint_keywords']:
            output += f"<li><b>Complaint Keywords:</b> {', '.join(result['entities']['complaint_keywords'])}</li>"
        else:
            output += "<li><b>Complaint Keywords:</b> None detected</li>"
        
        output += "</ul>"
        
        return output
    
    except Exception as e:
        return f"Error processing ticket: {str(e)}"

with gr.Blocks(title="Support Ticket Classifier & Entity Extractor") as demo:
    gr.Markdown("# Support Ticket Classifier & Entity Extractor")
    gr.Markdown("This application classifies support tickets by issue type and urgency level, and extracts entities like product names, dates, and complaint keywords.")
    
    with gr.Row():
        with gr.Column():
            train_button = gr.Button("Load and Train Model")
            status_output = gr.HTML("Model not loaded. Click 'Load and Train Model' to begin.")
            train_button.click(load_and_train_model, inputs=None, outputs=status_output)
    
    with gr.Row():
        with gr.Column():
            ticket_input = gr.Textbox(
                label="Enter Support Ticket Text",
                placeholder="Type or paste the support ticket text here...",
                lines=5
            )
            predict_button = gr.Button("Analyze Ticket")
        
        with gr.Column():
            result_output = gr.HTML(label="Analysis Results")
    
    predict_button.click(predict, inputs=ticket_input, outputs=result_output)
    
    gr.Markdown("## Example Tickets")
    
    example1 = "I purchased a SmartWatch V2 on January 15th, 2023, and it's been nothing but trouble. The screen keeps freezing and the battery drains within hours. This is the third time I've had issues with this product. I need a replacement as soon as possible, as I use this for work daily. My order number is #45678."
    example2 = "My RoboChef Blender arrived yesterday but is missing the instruction manual. Could you please email me a PDF copy? I'd like to use it this weekend for a family gathering."
    example3 = "URGENT: My Vision LED TV has completely stopped working after a power surge last night (05/28/2023). I have an important presentation for work tomorrow and need this fixed IMMEDIATELY. This is the second TV I've had issues with from your company. Very disappointed!"
    
    gr.Examples(
        examples=[example1, example2, example3],
        inputs=ticket_input
    )

if __name__ == "__main__":
    demo.launch()
