import pandas as pd
from sklearn.model_selection import StratifiedKFold
from PreProcess import PreProcess
from PreProcess.DataProcessor import DataProcessor
from Embeddings.GloveEmbeddingModel import GloveEmbeddingModel
from Embeddings.BERTEmbeddingModel import BERTEmbeddingModel
from DeepLearning.Seq2SeqLSTM import Seq2SeqLSTM
from DeepLearning.Seq2SeqGRU import Seq2SeqGRU
from DeepLearning.Seq2SeqGRUAttention import Seq2SeqGRUAttention
from DeepLearning. Seq2SeqTransformers import Seq2SeqTransformers
from ClassificationMetricsCalculator import ClassificationMetricsCalculator

# The experiments makes embedding technique constant
# and compare deep learning technique for each dataset.


# Load datasets using PreProcess class
abd_file_path = 'amazon_brand_dataset.csv'
chd_file_path = 'chicago_hotel_dataset.csv'
apd_file_path = 'amazon_petstore_dataset.csv'

preprocessor = PreProcess(abd_file_path, chd_file_path, apd_file_path)
abd_df, chd_df, apd_df = preprocessor.load_datasets()

# Define K-fold cross-validation
k = 5  # Number of folds
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Initialize metrics calculator
metrics_calculator = ClassificationMetricsCalculator()

# Define preprocessing techniques
preprocessing_techniques = ['stopword_removal', 'special_character_removal', 'lemmatization', 'stemming', 'ner']

# Create a dictionary to store results
results = []

# Iterate over datasets
datasets = [('ABD', abd_df), ('CHD', chd_df), ('APD', apd_df)]

for dataset_name, dataset in datasets:
    for prep_technique in preprocessing_techniques:
        # Preprocess dataset (fill missing data, remove outliers, etc.) using DataProcessor
        data_processor = DataProcessor(dataset)
        data_processor.fill_missing_data()
        data_processor.remove_outliers()

        if prep_technique == 'stopword_removal':
            data_processor.remove_stopwords()
        elif prep_technique == 'special_character_removal':
            data_processor.remove_special_characters()
        elif prep_technique == 'lemmatization':
            data_processor.lemmatize()
        elif prep_technique == 'stemming':
            data_processor.stem()
        elif prep_technique == 'ner':
            data_processor.apply_ner()

        # Initialize embeddings
        glove_model = GloveEmbeddingModel()
        glove_model.load('glove.6B.100d.txt')

        bert_model = BERTEmbeddingModel()

        # Apply different deep learning techniques
        deep_learning_models = [
            ('LSTM', Seq2SeqLSTM(glove_model)),
            ('GRU', Seq2SeqGRU(glove_model)),
            ('Attention', Seq2SeqGRUAttention(glove_model)),
            ('Transformers', Seq2SeqTransformers(bert_model))
        ]

        for deep_learning_name, deep_learning_model in deep_learning_models:
            fold_scores = []

            # Iterate over K-fold cross-validation
            for train_idx, test_idx in kf.split(dataset['text'], dataset['label']):
                X_train, X_test = dataset['text'].iloc[train_idx], dataset['text'].iloc[test_idx]
                y_train, y_test = dataset['label'].iloc[train_idx], dataset['label'].iloc[test_idx]

                # Train the deep learning model
                deep_learning_model.train(X_train, y_train)

                # Predict with the trained model
                y_pred = deep_learning_model.predict(X_test)

                # Evaluate the model using ClassificationMetricsCalculator
                metrics = metrics_calculator.compute_all_metrics(y_test, y_pred)

                # Store the evaluation metrics for each fold
                fold_scores.append(metrics)

            # Calculate and display the average metrics over all folds
            avg_metrics = {metric: sum([score[metric] for score in fold_scores]) / k for metric in metrics.keys()}
            results.append({
                'Dataset': dataset_name,
                'Preprocessing': prep_technique,
                'Deep Learning Technique': deep_learning_name,
                'Avg Metrics': avg_metrics
            })

# Print or analyze the results as needed
for result in results:
    print(result)
