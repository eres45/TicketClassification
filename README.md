# Support Ticket Classification and Entity Extraction

This project implements a machine learning pipeline for classifying support tickets by issue type and urgency level, as well as extracting key entities like product names, complaint keywords, and dates from ticket text.

## Project Overview

The system analyzes customer support tickets to automatically:

1. Classify the ticket by issue type (e.g., Product Defect, Billing Problem)
2. Determine the urgency level (Low, Medium, High)
3. Extract key entities from the text:
   - Product names
   - Dates mentioned
   - Complaint keywords

## Features

- **Data Preprocessing**: Clean, normalize, and tokenize text data
- **Feature Engineering**: TF-IDF vectorization with additional engineered features
- **Multiple Model Support**: Logistic Regression, SVM, and Random Forest classifiers
- **Entity Extraction**: Regex and dictionary-based entity extraction
- **Interactive UI**: Gradio web interface for real-time ticket analysis
- **Visualization**: Confusion matrices and classification reports

## Project Structure

- `ticket_classifier_robust.py`: Complete ML pipeline with training and evaluation
- `ticket_classifier_app.py`: Gradio app for interactive predictions
- `README.md`: Project documentation
- `output/`: Directory for generated visualizations and results
- `ai_dev_assignment_tickets_complex_1000.xls`: Dataset of support tickets

## Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- gradio
- openpyxl (for Excel file handling)

Install dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn gradio openpyxl
```

## Usage

### Training and Evaluating the Model

Run the ticket classifier:

```bash
python ticket_classifier_robust.py
```

This will:
1. Load and preprocess the data
2. Train multiple models for issue type and urgency level classification
3. Evaluate the models and generate confusion matrices
4. Test the models on example tickets

### Using the Interactive App

Launch the Gradio web interface:

```bash
python ticket_classifier_app.py
```

Then:
1. Click "Load and Train Model" to initialize the system
2. Enter a support ticket text in the input box
3. Click "Analyze Ticket" to get predictions
4. View the classification results and extracted entities

## Model Performance

The system achieves good performance through:

1. **Text Preprocessing**: Cleaning, normalization, and feature extraction
2. **Feature Engineering**:
   - TF-IDF vectorization captures important words and phrases
   - Additional features like ticket length and urgency term count
3. **Model Selection**: Automatically selects the best-performing model for each task

## Entity Extraction Approach

The system extracts entities using:

1. **Products**: Matches against a list of known products from the dataset
2. **Dates**: Regular expressions to identify various date formats
3. **Complaint Keywords**: Dictionary of common complaint terms

## Future Improvements

Potential enhancements include:

- Fine-tuning model hyperparameters
- Implementing more advanced NLP techniques (BERT, etc.)
- Adding more sophisticated entity extraction
- Expanding the complaint keyword dictionary
- Batch processing capabilities

## License

This project is developed as part of an assignment and is for educational purposes only.
