import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PreProcess import PreProcess
from PreProcess.DataProcessor import DataProcessor
from Embeddings.GloveEmbeddingModel import GloveEmbeddingModel
from Embeddings.BERTEmbeddingModel import BERTEmbeddingModel
from DeepLearning.Seq2SeqTransformers import Seq2SeqTransformers

# The experiments plots different data set for embedding technique
# Glove and BERT, deep learning technique transformers is
#constant, the plot is epoch versus accuracy.

# Load datasets using PreProcess class
abd_file_path = 'amazon_brand_dataset.csv'
chd_file_path = 'chicago_hotel_dataset.csv'
apd_file_path = 'amazon_petstore_dataset.csv'

preprocessor_abd = PreProcess(abd_file_path)
abd_df = preprocessor_abd.load_dataset()

preprocessor_chd = PreProcess(chd_file_path)
chd_df = preprocessor_chd.load_dataset()

preprocessor_apd = PreProcess(apd_file_path)
apd_df = preprocessor_apd.load_dataset()

# Define preprocessing techniques
preprocessing_techniques = ['stopword_removal', 'special_character_removal', 'lemmatization', 'stemming', 'ner']

# Define embedding techniques and their corresponding models
embedding_models = {
    'Glove': GloveEmbeddingModel(),
    'BERT': BERTEmbeddingModel()
}

# Define hyperparameters for the Seq2Seq model
seq2seq_hyperparameters = {
    'num_epochs': 10,     # Number of training epochs
    'batch_size': 32,    # Batch size
    'learning_rate': 0.001
}

# Initialize lists to store accuracy values for each dataset-embedding combination
results = []

# Iterate over datasets
datasets = [('ABD', abd_df), ('CHD', chd_df), ('APD', apd_df)]

for dataset_name, dataset in datasets:
    for prep_technique in preprocessing_techniques:
        # Preprocess dataset (stopword removal, special character removal, etc.) using DataProcessor
        data_processor = DataProcessor(dataset)
        data_processor.remove_stopwords()
        data_processor.remove_special_characters()
        data_processor.lemmatize()
        data_processor.stem()
        data_processor.apply_ner()

        # Initialize and train the Seq2Seq model for each embedding technique
        for emb_name, emb_model in embedding_models.items():
            X = emb_model.transform(data_processor.get_processed_text())

            # Initialize and train the Seq2Seq model
            seq2seq_model = Seq2SeqTransformers(seq2seq_hyperparameters)
            accuracy = seq2seq_model.train_and_evaluate(X, dataset['label'])

            # Store the accuracy for this dataset-embedding combination
            results.append({
                'Dataset': dataset_name,
                'Preprocessing': prep_technique,
                'Embedding': emb_name,
                'Accuracy': accuracy
            })

# Plot epoch versus accuracy graph for each dataset-embedding combination
plt.figure(figsize=(12, 12))
for i, dataset_name in enumerate(datasets):
    for j, emb_name in enumerate(embedding_models.keys()):
        plt.subplot(3, 2, i * 2 + j + 1)
        plt.title(f'{dataset_name} with {emb_name}')
        for result in results:
            if result['Dataset'] == dataset_name and result['Embedding'] == emb_name:
                plt.plot(range(1, seq2seq_hyperparameters['num_epochs'] + 1), result['Accuracy'], label=result['Preprocessing'])

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend('Bert+ABD','Glove+ABD','Bert+CHD','Glove+CHD','Bert+APD','Glove+APD')

plt.tight_layout()
plt.show()
