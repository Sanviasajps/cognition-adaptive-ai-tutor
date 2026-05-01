class SimpleTokenizer:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}

    def build_vocab(self, texts):
        words = set()

        for text in texts:
            words.update(text.split())

        self.word2id = {w: i for i, w in enumerate(words)}
        self.id2word = {i: w for w, i in self.word2id.items()}

    def encode(self, text):
        return [self.word2id[w] for w in text.split() if w in self.word2id]

    def decode(self, ids):
        return " ".join([self.id2word[i] for i in ids])