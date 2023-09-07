import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

class PreProcess:
    def __init__(self, a_bd_file_path, c_hd_file_path, a_pd_file_path):
        self.a_bd_file_path = a_bd_file_path
        self.c_hd_file_path = c_hd_file_path
        self.a_pd_file_path = a_pd_file_path
        self.stopwords = set(stopwords.words('english'))
        self.nlp = spacy.load('en_core_web_sm')
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def load_datasets(self):
        # Load the three datasets into Pandas DataFrames
        self.abd_df = pd.read_csv(self.a_bd_file_path)
        self.chd_df = pd.read_csv(self.c_hd_file_path)
        self.apd_df = pd.read_csv(self.a_pd_file_path)

    def remove_stopwords(self, text):
        # Tokenize the text and remove stopwords
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in self.stopwords]
        return ' '.join(filtered_tokens)

    def lemmatize_text(self, text):
        # Lemmatize the text
        return ' '.join([self.lemmatizer.lemmatize(word) for word in word_tokenize(text)])

    def stem_text(self, text):
        # Stem the text
        return ' '.join([self.stemmer.stem(word) for word in word_tokenize(text)])

    def perform_ner(self, text):
        # Perform Named Entity Recognition (NER) using spaCy
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents]
        return ' '.join(entities)

    def remove_special_characters(self, text):
        # Remove special characters and non-alphanumeric characters
        return ''.join(e for e in text if e.isalnum() or e.isspace())

    def preprocess_datasets(self):
        # Preprocess text in each dataset
        self.abd_df['processed_text'] = self.abd_df['text'].apply(self.remove_special_characters)
        self.abd_df['processed_text'] = self.abd_df['processed_text'].apply(self.remove_stopwords)
        self.abd_df['processed_text'] = self.abd_df['processed_text'].apply(self.lemmatize_text)

        self.chd_df['processed_text'] = self.chd_df['text'].apply(self.remove_special_characters)
        self.chd_df['processed_text'] = self.chd_df['processed_text'].apply(self.remove_stopwords)
        self.chd_df['processed_text'] = self.chd_df['processed_text'].apply(self.stem_text)

        self.apd_df['processed_text'] = self.apd_df['text'].apply(self.remove_special_characters)
        self.apd_df['processed_text'] = self.apd_df['processed_text'].apply(self.remove_stopwords)
        self.apd_df['processed_text'] = self.apd_df['processed_text'].apply(self.perform_ner)

    def get_processed_datasets(self):
        # Return the processed datasets
        return self.abd_df, self.chd_df, self.apd_df
