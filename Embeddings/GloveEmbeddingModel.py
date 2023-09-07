from gensim.models import KeyedVectors

class GloveEmbeddingModel:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.model = None

    def load(self):
        if self.filepath:
            self.model = KeyedVectors.load_word2vec_format(self.filepath, binary=False)

    def get_word_vector(self, word):
        if self.model and word in self.model:
            return self.model[word]
        else:
            return None

    def most_similar_words(self, word, topn=10):
        if self.model and word in self.model:
            return self.model.most_similar(word, topn=topn)
        else:
            return []

    def save(self, filepath):
        if self.model:
            self.model.save_word2vec_format(filepath, binary=False)

