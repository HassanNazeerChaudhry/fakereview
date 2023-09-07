import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ClassificationMetricsCalculator import ClassificationMetricsCalculator
from PreProcess import PreProcess
from PreProcess.DataProcessor import DataProcessor
from Embeddings.TFIDFVectorizer import TFIDFVectorizer
from Embeddings.BigramTrigramExtractor import BigramTrigramExtractor
from Embeddings.Word2VecModel import Word2VecModel
from Embeddings.GloveEmbeddingModel import GloveEmbeddingModel
from Embeddings.BERTEmbeddingModel import BERTEmbeddingModel

# Load the datasets using the PreProcess class
abd_file_path = 'amazon_brand_dataset.csv'
chd_file_path = 'chicago_hotel_dataset.csv'
apd_file_path = 'amazon_petstore_dataset.csv'



preprocessor = PreProcess(abd_file_path, chd_file_path, apd_file_path)
preprocessor.load_datasets()

# Check if datasets are balanced
abd_balanced = preprocessor.check_balance()
chd_balanced = preprocessor.check_balance()
apd_balanced = preprocessor.check_balance()

# Perform K-fold cross-validation
k = 5  # Number of folds
abd_scores = []
chd_scores = []
apd_scores = []

for train_index, test_index in StratifiedKFold(n_splits=k).split(preprocessor.abd_df, preprocessor.abd_df['label']):
    train_abd, test_abd = preprocessor.abd_df.iloc[train_index], preprocessor.abd_df.iloc[test_index]
    data_processor = DataProcessor(train_abd)

    # Create a DataProcessor instance for preprocessing
    data_processor = DataProcessor(train_abd)

    # Fill missing data using DataProcessor
    data_processor.fill_missing_data()

    # Remove outliers using DataProcessor (adjust the threshold as needed)
    data_processor.remove_outliers(threshold=3)


    # Apply TF-IDF vectorization
    tfidf_vectorizer = TFIDFVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(data_processor.get_processed_text())

    # Apply Bigram and Trigram extraction
    bigram_trigram_extractor = BigramTrigramExtractor()
    X_train_bigram = bigram_trigram_extractor.extract_bigrams(train_abd['processed_text'])
    X_train_trigram = bigram_trigram_extractor.extract_trigrams(train_abd['processed_text'])

    # Apply Word2Vec embedding
    word2vec_model = Word2VecModel()
    word2vec_model.train(X_train_bigram + X_train_trigram)

    # Apply GloVe embedding
    glove_model = GloveEmbeddingModel()
    glove_model.load('glove.6B.100d.txt')

    # Apply BERT embedding
    bert_model = BERTEmbeddingModel()
    train_text = train_abd['processed_text'].tolist()
    X_train_bert = bert_model.get_bert_embeddings(train_text)

    # Apply classification models
    nb_classifier = MultinomialNB()
    svc_classifier = SVC()
    linear_svc_classifier = LinearSVC()
    rf_classifier = RandomForestClassifier()
    xgb_classifier = XGBClassifier()

    # Train and evaluate classifiers
    classifiers = [nb_classifier, svc_classifier, linear_svc_classifier, rf_classifier, xgb_classifier]
    for classifier in classifiers:
        classifier.fit(X_train_tfidf, train_abd['label'])
        y_pred = classifier.predict(tfidf_vectorizer.transform(test_abd['processed_text']))
        metrics_calculator = ClassificationMetricsCalculator(test_abd['label'], y_pred)
        metrics = metrics_calculator.compute_all_metrics()
        abd_scores.append(metrics)

# Repeat the same steps for CHD and APD datasets

# Print or analyze the results as needed
for i, metrics in enumerate(abd_scores):
    print(f"Classifier {i + 1} Metrics for ABD:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print()
