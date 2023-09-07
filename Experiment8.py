import pandas as pd
import matplotlib.pyplot as plt
from PreProcess import PreProcess
from PreProcess.DataProcessor import DataProcessor
from Embeddings.BERTEmbeddingModel import BERTEmbeddingModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the Amazon Brand dataset using PreProcess class
abd_file_path = 'amazon_brand_dataset.csv'

preprocessor = PreProcess(abd_file_path)
abd_df = preprocessor.load_dataset()

# Define preprocessing techniques
preprocessing_techniques = ['stopword_removal', 'special_character_removal', 'lemmatization', 'stemming', 'ner']

# Define the embedding model
embedding_model = BERTEmbeddingModel()

# Split the dataset into train and test sets
X = embedding_model.transform(abd_df['text'])
y = abd_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Naive Bayes': MultinomialNB(),
    'Linear SVM': LinearSVC(),
    'SVM with Grid Search': GridSearchCV(SVC(), param_grid={'C': [1, 10, 100]}, cv=5),
    'XG Boost': XGBClassifier()
}

# Train and evaluate models
results = []

for prep_technique in preprocessing_techniques:
    data_processor = DataProcessor(abd_df)
    data_processor.remove_stopwords()
    data_processor.remove_special_characters()
    data_processor.lemmatize()
    data_processor.stem()
    data_processor.apply_ner()

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results.append({
            'Model': model_name,
            'Preprocessing': prep_technique,
            'Accuracy': accuracy
        })

# Plot accuracy comparison graph for different models
plt.figure(figsize=(10, 6))
for model_name in models.keys():
    plt.title('Accuracy Comparison for Different Models')
    model_accuracies = [result['Accuracy'] for result in results if result['Model'] == model_name]
    preprocessing_labels = [result['Preprocessing'] for result in results if result['Model'] == model_name]

    plt.bar(preprocessing_labels, model_accuracies, label=model_name)

plt.xlabel('Preprocessing Technique')
plt.ylabel('Accuracy')
plt.legend('NB','NB Grid Search','Linear SVM', 'SVM with grid search','XGBoost')
plt.show()
