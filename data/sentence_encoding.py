import numpy as np


class SentenceEncoding:

    def __init__(self, vocab, delimiter=' '):
        self.vocab = vocab
        self.delimiter = delimiter

    def encode_batch(self, sentence_batch, padding):
        encoded = []
        for s in sentence_batch:
            encoded.append(self.encode(s, padding))
        return np.array(encoded)

    def encode(self, sentence, padding):
        tokens = sentence.split(self.delimiter)
        size = len(tokens)
        tokens = tokens + max(0, padding - size) * [self.vocab.end_token]
        return np.array([self.vocab.encode(token) for token in tokens])

    def decode(self, tokens):
        def decode_gen():
            for t in tokens:
                decoded = self.vocab.decode(t)
                if decoded == self.vocab.end_token:
                    return
                yield decoded

        return self.delimiter.join(list(decode_gen()))

    def decode_batch(self, tokens_batch):
        decoded = []
        for tokens in tokens_batch:
            decoded.append(self.decode(tokens))
        return decoded
