from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

class Word2VecModel:
    def __init__(self, sentences=None, vector_size=100, window=5, min_count=1, sg=0):
        self.sentences = sentences
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.model = None

    def train(self):
        if self.sentences:
            tokenized_sentences = [word_tokenize(sentence) for sentence in self.sentences]
            self.model = Word2Vec(
                tokenized_sentences,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                sg=self.sg
            )
            self.model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=10)

    def get_word_vector(self, word):
        if self.model and word in self.model.wv:
            return self.model.wv[word]
        else:
            return None

    def most_similar_words(self, word, topn=10):
        if self.model and word in self.model.wv:
            return self.model.wv.most_similar(word, topn=topn)
        else:
            return []

    def save(self, filepath):
        if self.model:
            self.model.save(filepath)

    def load(self, filepath):
        self.model = Word2Vec.load(filepath)
