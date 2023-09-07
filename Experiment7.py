import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PreProcess import PreProcess
from PreProcess.DataProcessor import DataProcessor
from Embeddings.BERTEmbeddingModel import BERTEmbeddingModel
from DeepLearning.Seq2SeqLSTM import Seq2SeqLSTM
from DeepLearning.Seq2SeqGRU import Seq2SeqGRU
from DeepLearning.Seq2SeqLSTMAttension import Seq2SeqLSTMAttension
from DeepLearning.Seq2SeqGRUAttention import Seq2SeqGRUAttention
from DeepLearning.Seq2SeqTransformers import Seq2SeqTransformers

# Load the Amazon Brand dataset using PreProcess class
abd_file_path = 'amazon_brand_dataset.csv'

preprocessor = PreProcess(abd_file_path)
abd_df = preprocessor.load_dataset()

# Define preprocessing techniques
preprocessing_techniques = ['stopword_removal', 'special_character_removal', 'lemmatization', 'stemming', 'ner']

# Define the embedding model
embedding_model = BERTEmbeddingModel()

# Define hyperparameters for the Seq2Seq models
seq2seq_hyperparameters = {
    'num_epochs': 10,     # Number of training epochs
    'batch_size': 32,    # Batch size
    'learning_rate': 0.001
}

# Initialize lists to store accuracy values for each model
results = []

# Define sequence-to-sequence models
seq2seq_models = [
    ('LSTM', Seq2SeqLSTM(embedding_model, seq2seq_hyperparameters)),
    ('GRU', Seq2SeqGRU(embedding_model, seq2seq_hyperparameters)),
    ('LSTM + Attention', Seq2SeqLSTMAttension(embedding_model, seq2seq_hyperparameters)),
    ('GRU + Attention', Seq2SeqGRUAttention(embedding_model, seq2seq_hyperparameters)),
    ('Transformers', Seq2SeqTransformers(embedding_model, seq2seq_hyperparameters))
]

# Iterate over preprocessing techniques
for prep_technique in preprocessing_techniques:
    # Preprocess dataset (stopword removal, special character removal, etc.) using DataProcessor
    data_processor = DataProcessor(abd_df)
    data_processor.remove_stopwords()
    data_processor.remove_special_characters()
    data_processor.lemmatize()
    data_processor.stem()
    data_processor.apply_ner()

    # Initialize and train each sequence-to-sequence model
    for model_name, seq2seq_model in seq2seq_models:
        X = data_processor.get_processed_text()

        # Train the Seq2Seq model
        accuracy = seq2seq_model.train_and_evaluate(X, abd_df['label'])

        # Store the accuracy for this model
        results.append({
            'Model': model_name,
            'Preprocessing': prep_technique,
            'Accuracy': accuracy
        })

# Plot epoch versus accuracy graph for each sequence-to-sequence model
plt.figure(figsize=(10, 6))
for model_name in [model[0] for model in seq2seq_models]:
    plt.title(f'{model_name}')
    for result in results:
        if result['Model'] == model_name:
            plt.plot(range(1, seq2seq_hyperparameters['num_epochs'] + 1), result['Accuracy'], label=result['Preprocessing'])

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend('LSTM','GRU','LSTM+Attension','GRU+Attension', 'Transformers')

plt.tight_layout()
plt.show()
