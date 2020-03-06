import numpy


class FilteredFastText(object):
    """
    Stores the word vectors only present on vocabulary
    """
    def __init__(self, vocabulary, fast_text_model):
        self.vocabulary = vocabulary
        self.fast_text_model = fast_text_model

        self.word2index = {}
        self.wv = numpy.zeros(
                    (len(self.vocabulary), self.fast_text_model.vector_size)
                )

        for index, word in enumerate(self.vocabulary):
            if word in self.fast_text_model.vocab:  # Add the embedding
                self.wv[index, :] = self.fast_text_model[word]
            else:  # Add a random vector!
                self.wv[index, :] = numpy.random.randn(
                                            self.fast_text_model.vector_size
                                    )
            self.word2index[word] = index

    def get_vector(self, word):
        if word in self.word2index:
            return self.wv[self.word2index[word]]
        else:  # Random vector
            return numpy.random.randn(self.fast_text_model.wv.vector_size)
