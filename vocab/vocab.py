class Vocabulary:

    def __init__(self, words):
        self._word_to_int = dict([(w, i) for i, w in enumerate(words)])
        self._int_to_word = dict([(i, w) for i, w in enumerate(words)])
        self.words = words
        self.unknown_token = words[0]
        self.start_token = words[1]
        self.end_token = words[2]
        self.size = len(words)

    def encode(self, word):
        if word in self._word_to_int:
            return self._word_to_int[word]
        else:
            return self._word_to_int[self.unknown_token]

    def decode(self, val):
        if val < self.size:
            return self._int_to_word[val]
        else:
            raise Exception('word given by index {} not found'.format(val))

    def write_to_file(self, filename):
        with open(filename, 'w') as f:
            for w in self.words:
                print(w)
                f.write(w + '\n')

    @staticmethod
    def read_from_file(filename):
        words = []
        with open(filename) as f:
            for line in f:
                word = line.replace('\n', '')
                words.append(word)
        return Vocabulary(words)

    @staticmethod
    def create_from_file(filename, vocab_size=2000, start_token='<s>', end_token='</s>', unknown_token='<unk>'):
        words = {}
        with open(filename) as f:
            for line in f:
                line = line.replace('\n', '')
                tokens = line.split(' ')
                for word in tokens:
                    if word in words:
                        words[word] = 1 + words[word]
                    else:
                        words[word] = 1
        return Vocabulary([unknown_token, start_token, end_token] + list(
            map(lambda x: x[0], sorted(words.items(), key=lambda x: x[1], reverse=True)))[:vocab_size - 3])