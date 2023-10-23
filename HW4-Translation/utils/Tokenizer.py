import pickle

class ch_Tokenizer:
    def __init__(self, dict_path):
        with open(dict_path+'ch.pkl', "rb") as file:
            ch2index = pickle.load(file)
            file.close()

        word2index = {word: index + 3 for word, index in ch2index.items()}
        word2index.update({"<PAD>": 0, "<BOS>": 1, "<EOS>": 2})
        index2word = {v: k for k, v in word2index.items()}

        self.word2index = word2index
        self.index2word = index2word
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2

    def encode(self, sentence):
        out = []
        for w in sentence:
            if w not in self.word2index.keys():
                pass
            else:
                out.append(self.word2index[w])
        return out

    def decode(self, indexes):
        return [self.index2word[index] for index in indexes]

    def length(self):
        return len(self.index2word)

class tl_Tokenizer:
    def __init__(self, dict_path):
        with open(dict_path+'tl.pkl', "rb") as file:
            tl2index = pickle.load(file)
            file.close()

        word2index = {word: index + 3 for word, index in tl2index.items()}
        word2index.update({"<PAD>": 0, "<BOS>": 1, "<EOS>": 2})
        index2word = {v: k for k, v in word2index.items()}

        self.word2index = word2index
        self.index2word = index2word
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2

    def encode(self, sentence):
        out = []
        for w in sentence:
            if w not in self.word2index.keys():
                pass
            else:
                out.append(self.word2index[w])
        return out

    def decode(self, indexes):
        return [self.index2word[index] for index in indexes]

    def decode_single(self,indexes):
        return self.index2word[indexes]

    def length(self):
        return len(self.index2word)