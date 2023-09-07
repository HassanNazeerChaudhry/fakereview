
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from PreProcess import PreProcess
from PreProcess.DataProcessor import DataProcessor
from Embeddings.TFIDFVectorizer import TFIDFVectorizer
from Embeddings.BigramTrigramExtractor import BigramTrigramExtractor
from Embeddings.Word2VecModel import Word2VecModel
from Embeddings.GloveEmbeddingModel import GloveEmbeddingModel
from Embeddings.BERTEmbeddingModel import BERTEmbeddingModel
from DeepLearning.Seq2SeqLSTM import Seq2SeqLSTM
from DeepLearning.Seq2SeqGRU import Seq2SeqGRU
from DeepLearning.Seq2SeqGRUAttention import Seq2SeqGRUAttention
from DeepLearning.Seq2SeqTransformers import Seq2SeqTransformers
from ClassificationMetricsCalculator import ClassificationMetricsCalculator

# The experiments compare deep learning technique
#as well as different embedding techniques
#for each dataset.

# Load datasets using PreProcess class
abd_file_path = 'amazon_brand_dataset.csv'
chd_file_path = 'chicago_hotel_dataset.csv'
apd_file_path = 'amazon_petstore_dataset.csv'

preprocessor = PreProcess(abd_file_path, chd_file_path, apd_file_path)
abd_df, chd_df, apd_df = preprocessor.load_datasets()

# Check if datasets are balanced
abd_balanced = preprocessor.check_balance(abd_df)
chd_balanced = preprocessor.check_balance(chd_df)
apd_balanced = preprocessor.check_balance(apd_df)

# Define K-fold cross-validation
k = 5  # Number of folds
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Initialize metrics calculators
metrics_calculator = ClassificationMetricsCalculator()

# Define preprocessing and embedding techniques
preprocessing_techniques = ['stopword_removal', 'special_character_removal', 'lemmatization', 'stemming', 'ner']
embedding_techniques = ['TF-IDF', 'BigramTrigram', 'Word2Vec', 'Glove', 'BERT']

# Iterate over datasets and techniques
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

        # Apply different embedding techniques
        for emb_technique in embedding_techniques:
            if emb_technique == 'TF-IDF':
                tfidf_vectorizer = TFIDFVectorizer()
                X = tfidf_vectorizer.fit_transform(data_processor.get_processed_text())
            elif emb_technique == 'BigramTrigram':
                bigram_trigram_extractor = BigramTrigramExtractor()
                X = bigram_trigram_extractor.extract_bigrams_trigrams(data_processor.get_processed_text())
            elif emb_technique == 'Word2Vec':
                word2vec_model = Word2VecModel()
                word2vec_model.train(data_processor.get_processed_text())
                X = word2vec_model.transform(data_processor.get_processed_text())
            elif emb_technique == 'Glove':
                glove_model = GloveEmbeddingModel()
                glove_model.load('glove.6B.100d.txt')
                X = glove_model.transform(data_processor.get_processed_text())
            elif emb_technique == 'BERT':
                bert_model = BERTEmbeddingModel()
                X = bert_model.get_bert_embeddings(data_processor.get_processed_text())

                # Initialize sequence-to-sequence models
                seq2seq_lstm = Seq2SeqLSTM(
                    max_sequence_length=128,  # Specify the maximum sequence length
                    vocab_size=10000,  # Specify the vocabulary size
                    embedding_dim=256,  # Specify the embedding dimension
                    lstm_units=512  # Specify the LSTM units
                )

                seq2seq_gru = Seq2SeqGRU(
                    max_sequence_length=128,
                    vocab_size=10000,
                    embedding_dim=256,
                    gru_units=512
                )

                seq2seq_attention = Seq2SeqGRUAttention(
                    max_sequence_length=128,
                    vocab_size=10000,
                    embedding_dim=256,
                    gru_units=512
                )

                mdl_path = 'transformers_model'
                seq2seq_transformers = Seq2SeqTransformers(
                    max_sequence_length=128,
                    tokenizer_path='bert-base-uncased',  # Specify the tokenizer path
                    model_path=mdl_path  # Specify the pre-trained model path
                )
            # Iterate over sequence-to-sequence models
            seq2seq_models = [
                ('LSTM', seq2seq_lstm),
                ('GRU',seq2seq_gru),
                ('Attention', seq2seq_attention),
                ('Transformers', seq2seq_transformers)
            ]

            for seq2seq_name, seq2seq_model in seq2seq_models:
                fold_scores = []

                # Iterate over K-fold cross-validation
                for train_idx, test_idx in kf.split(X, dataset['label']):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = dataset['label'].iloc[train_idx], dataset['label'].iloc[test_idx]

                    # Train the sequence-to-sequence model
                    seq2seq_model.train(X_train, y_train)

                    # Predict with the trained model
                    y_pred = seq2seq_model.predict(X_test)

                    # Evaluate the model using ClassificationMetricsCalculator
                    metrics = metrics_calculator.compute_all_metrics(y_test, y_pred)

                    # Store the evaluation metrics for each fold
                    fold_scores.append(metrics)

                # Calculate and display the average metrics over all folds
                avg_metrics = {metric: np.mean([score[metric] for score in fold_scores]) for metric in metrics.keys()}
                print(f'Dataset: {dataset_name}, Preprocessing: {prep_technique}, '
                      f'Embedding: {emb_technique}, Seq2Seq Model: {seq2seq_name}')
                print(f'Average Metrics over {k}-fold Cross-validation:')
                for metric, value in avg_metrics.items():
                    print(f'{metric}: {value:.4f}')
                print('-' * 50)

