# Tokenizador de Tweets

"""
> import nltk
> nltk.download('punkt')
"""
import nltk.tokenize as tknz
from spacy.tokenizer import Tokenizer
from preprocessing import PreprocessingReview


class ReviewTokenizer:

    def __init__(self, punctuation=False, change_words=False, stem=False):
        self.punctuation = punctuation
        self.change_words = change_words
        self.stem = stem

        # Stopwords del ingles
        self.preprocessing = PreprocessingReview()
        self.english_stopwords = self.preprocessing.get_english_stopwords()

    def __call__(self, X):
        """
        # http://juanpabloaj.com/2016/08/06/Algunos-metodos-especiales-de-clase/
        """
        tokens = tknz.word_tokenize(X)

        if self.punctuation:
            tokens = [t for t in tokens
                      if not self.preprocessing.is_punctuation(t)]
            tokens = [t for t in tokens
                      if not self.preprocessing.is_empty_string(t)]

        if self.change_words:
            tokens = [self.preprocessing.change_pos_neg_word(t)
                      for t in tokens]

        if self.stem:
            tokens = [self.preprocessing.text_stemming(t) for t in tokens]

        return tokens
