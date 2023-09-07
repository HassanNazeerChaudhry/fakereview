import nltk
from nltk.util import bigrams, trigrams

class BigramTrigramExtractor:
    def __init__(self):
        nltk.download('punkt')  # Download the punkt tokenizer data (if not already downloaded)

    def extract_bigrams(self, text):
        # Tokenize the text and generate bigrams
        tokens = nltk.word_tokenize(text)
        return list(bigrams(tokens))

    def extract_trigrams(self, text):
        # Tokenize the text and generate trigrams
        tokens = nltk.word_tokenize(text)
        return list(trigrams(tokens))
