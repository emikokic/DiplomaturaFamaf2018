import re
import string
import unidecode
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


class PreprocessingReview:

    def __init__(self):
        # Stopwords del español
        # Descargar las stopwords en una consola de Python, haciendo:
        # >>> import nltk
        # >>> nltk.download()
        self.english_stopwords = stopwords.words('english')

        # Lematizador del español
        self.stemmer = SnowballStemmer('english')

        # len(pos_words) = 2301
        self.pos_words = self.get_MPQA_pos_words()

        # len(neg_words) = 4151
        self.neg_words = self.get_MPQA_neg_words()

    def get_english_stopwords(self):
        """
        Retorna las Stopwords del Ingles.
        """
        return self.english_stopwords

    def load_MPQA_lexicon(self):
        with open('../subjclueslen1-HLTEMNLP05.tff') as f:
            lines = f.readlines()

        words = []
        for line in lines:
            sline = line.split()
            dline = dict([token.split('=') for token in sline if '=' in token])
            word = dline['word1']
            pol = dline['priorpolarity']

            if pol not in {'both', 'neutral'}:
                if pol in {'negative', 'weakneg'}:
                    pol = 'NEG'
                else:
                    pol = 'POS'

                words.append((word, pol))

        word_dict = dict(words)

        return word_dict

    def get_MPQA_pos_words(self):
        word_dict = self.load_MPQA_lexicon()

        return [k for k, v in word_dict.items() if v == 'POS']

    def get_MPQA_neg_words(self):
        word_dict = self.load_MPQA_lexicon()

        return [k for k, v in word_dict.items() if v == 'NEG']

    def change_pos_neg_word(self, word):
        """
        Reemplaza todas las palabras positivas y negativas presentes en los
        conjuntos "self.pos_words" y "self.neg_words" por las palabras
        "POS_WORD" y "NEG_WORD" respectivamente.
        """
        if word in self.pos_words:
            return "POS_WORD"
        elif word in self.neg_words:
            return "NEG_WORD"
        else:
            return word

    def is_empty_string(self, word):
        """
        Detecta si un string es vacio.
        """
        return word == ''

    def is_punctuation(self, word):
        """
        Detecta si un string en un signo de puntuacion.
        """
        punctuation = set(string.punctuation)

        return word in punctuation

    def is_number(self, word):
        """
        Detecta si un string es un numero (digito).
        """
        digits = re.compile(r'\d+')
        return digits.match(word)

    def remove_accents(self, word):
        """
        Remplazamos todos los caracteres de una cadena por su version sin
        acentuar (sin tilde).

        Ejemplo:
            > remove_accents("mùndó")
            > mundo
        """
        return unidecode.unidecode(word)

    def text_stemming(self, word):
        """
        Retorna el Stemming de un string.
        Stemming es el proceso por el cual transformamos una palabra a su raiz.
        """
        reserved_words = {"POS_WORD", "NEG_WORD"}
        if word in reserved_words:
            return word
        else:
            return self.stemmer.stem(word)

    def handle_negations(self, tokens):
        """
        Al encontrar una negacion (en negative_adverbs) modifica todas las
        palabras hasta el siguiente signo de puntuacion, agregandole:
            * Prefijo NOT_
        """
        # http://www.ejemplos.co/30-ejemplos-de-adverbios-de-negacion/
        negative_adverbs = {
            "neither",
            "never",
            "no",
            "not"
        }

        punct_re = re.compile(r"(?u)[!(),.-:;?]+")
        new_tokens = []
        negate = False
        for t in tokens:
            if t in negative_adverbs:
                negate = True
            elif punct_re.match(t):
                negate = False
            elif negate:
                t = 'NOT_' + t
            new_tokens.append(t)

        return new_tokens

    def reduce_lengthening(self, word, length=1):
        """
        Reemplaza secuencias de caracteres repetidos de longitud 3 o mayor
        con secuencias de longitud 1, 2 o 3.
        """
        pattern = re.compile(r"(.)\1{2,}")
        if length == 1:
            return pattern.sub(r"\1", word)
        elif length == 2:
            return pattern.sub(r"\1\1", word)
        elif length == 3:
            return pattern.sub(r"\1\1\1", word)
