from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFVectorizer:
    def __init__(self, max_features=None, stop_words=None):
        self.max_features = max_features
        self.stop_words = stop_words
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words=self.stop_words)

    def fit_transform(self, documents):
        # Fit the TF-IDF vectorizer to the documents and transform them
        return self.vectorizer.fit_transform(documents)

    def transform(self, documents):
        # Transform new documents using the pre-fit vectorizer
        return self.vectorizer.transform(documents)

    def get_feature_names(self):
        # Get the feature (term) names
        return self.vectorizer.get_feature_names_out()
