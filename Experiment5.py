import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PreProcess import PreProcess
from PreProcess.DataProcessor import DataProcessor
from Embeddings.TFIDFVectorizer import TFIDFVectorizer
from Embeddings.BigramTrigramExtractor import BigramTrigramExtractor
from Embeddings.Word2VecModel import Word2VecModel
from Embeddings.GloveEmbeddingModel import GloveEmbeddingModel
from Embeddings.BERTEmbeddingModel import BERTEmbeddingModel
from DeepLearning.Seq2SeqTransformers import Seq2SeqTransformers


# The experiments plots different embedding technique
# by making dataset ABD and deep learning technique transformers
#constant, the plot is epoch versus accuracy.

# Load the Amazon Brand dataset using PreProcess class
abd_file_path = 'amazon_brand_dataset.csv'

preprocessor = PreProcess(abd_file_path)
abd_df = preprocessor.load_dataset()

# Define preprocessing techniques
preprocessing_techniques = ['stopword_removal', 'special_character_removal', 'lemmatization', 'stemming', 'ner']

# Define embedding techniques and their corresponding models
embedding_models = {
    'TF-IDF': TFIDFVectorizer(),
    'BigramTrigram': BigramTrigramExtractor(),
    'Word2Vec': Word2VecModel(),
    'Glove': GloveEmbeddingModel(),
    'BERT': BERTEmbeddingModel()
}

# Define hyperparameters for the Seq2Seq model
seq2seq_hyperparameters = {
    'num_epochs': 10,     # Number of training epochs
    'batch_size': 32,    # Batch size
    'learning_rate': 0.001
}

# Initialize lists to store accuracy values for each embedding technique
embedding_accuracies = {emb_name: [] for emb_name in embedding_models.keys()}

# Iterate over preprocessing techniques
for prep_technique in preprocessing_techniques:
    # Preprocess dataset (stopword removal, special character removal, etc.) using DataProcessor
    data_processor = DataProcessor(abd_df)
    data_processor.remove_stopwords()
    data_processor.remove_special_characters()
    data_processor.lemmatize()
    data_processor.stem()
    data_processor.apply_ner()

    # Iterate over embedding techniques
    for emb_name, emb_model in embedding_models.items():
        X = emb_model.transform(data_processor.get_processed_text())

        # Initialize and train the Seq2Seq model
        seq2seq_model = Seq2SeqTransformers(seq2seq_hyperparameters)
        accuracy = seq2seq_model.train_and_evaluate(X, abd_df['label'])

        # Store the accuracy for this combination of preprocessing and embedding
        embedding_accuracies[emb_name].append(accuracy)

# Plot epoch versus accuracy graph for each embedding technique
epochs = np.arange(1, seq2seq_hyperparameters['num_epochs'] + 1)

plt.figure(figsize=(10, 6))
for emb_name, accuracies in embedding_accuracies.items():
    plt.plot(epochs, accuracies, label=emb_name)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Epoch vs. Accuracy for Different Embedding Techniques')
plt.legend('TF-IDF','Bi-Gram','Tri-Grams','BoWs','Word2Vec','Bert', 'Glove')
plt.grid(True)
plt.show()
