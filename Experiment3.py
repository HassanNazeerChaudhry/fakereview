import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from PreProcess import PreProcess
from PreProcess.DataProcessor import DataProcessor
from Embeddings.TFIDFVectorizer import TFIDFVectorizer
from Embeddings.BigramTrigramExtractor import BigramTrigramExtractor
from Embeddings.Word2VecModel import Word2VecModel
from Embeddings.GloveEmbeddingModel import GloveEmbeddingModel
from Embeddings.BERTEmbeddingModel import BERTEmbeddingModel
from TransformersMultiheadAttention import TransformersMultiheadAttention
from ClassificationMetricsCalculator import ClassificationMetricsCalculator

# The experiments makes deep learning technique
#constant and compare different embedding techniques
#for each dataset.

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

# Define preprocessing and embedding techniques
preprocessing_techniques = ['stopword_removal', 'special_character_removal', 'lemmatization', 'stemming', 'ner']
embedding_techniques = ['TF-IDF', 'BigramTrigram', 'Word2Vec', 'Glove', 'BERT']

# Create a dictionary to store results
results = []

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

            # Apply Transformers Multi-head Attention on the embeddings
            transformers_attention = TransformersMultiheadAttention()
            transformed_X = transformers_attention.apply_attention(X)

            # Perform classification (you can use any classifier)
            from DeepLearning.Seq2SeqGRU import Seq2SeqGRU  # Import your GRU-based model

            fold_scores = []

            for train_idx, test_idx in kf.split(transformed_X, dataset['label']):
                X_train, X_test = transformed_X[train_idx], transformed_X[test_idx]
                y_train, y_test = dataset['label'].iloc[train_idx], dataset['label'].iloc[test_idx]

                # Initialize and train the GRU-based sequence-to-sequence model
                gru_model = Seq2SeqGRU(
                    max_sequence_length=X_train.shape[1],  # Replace with your sequence length
                    vocab_size=X_train.shape[2],  # Replace with your vocab size
                    gru_units=128,  # Adjust the GRU units as needed
                )

                gru_model.train(X_train, y_train)
                y_pred = gru_model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                fold_scores.append(accuracy)

            # Calculate the average accuracy over all folds
            avg_accuracy = sum(fold_scores) / k

            # Store the results in the dictionary
            results.append({
                'Dataset': dataset_name,
                'Preprocessing': prep_technique,
                'Embedding': emb_technique,
                'Avg Accuracy': avg_accuracy
            })

            # ...

            # Print or analyze the results as needed
            for result in results:
                print(result)
