from lib2to3.pgen2 import token
import pickle
from collections import Counter
import json
from re import I
import jieba
import re


class JsonReader(object):
    def __init__(self, json_file):
        self.data = self.__read_json(json_file)
        self.keys = list(self.data.keys())

    def __read_json(self, filename):
        with open(filename, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        return data

    def __getitem__(self, item):
        return self.data[item]
        # return self.data[self.keys[item]]

    def __len__(self):
        return len(self.data)


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.id2word = {}
        self.idx = 0
        self.add_word('<pad>')
        self.add_word('<end>')
        self.add_word('<start>')
        self.add_word('<unk>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.id2word[self.idx] = word
            self.idx += 1

    def get_word_by_id(self, id):
        return self.id2word[id]

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json_file, threshold):
    caption_reader = JsonReader(json_file)
    counter = Counter()

    for items in caption_reader:
        # text = items.replace('', '').replace(',', '')
        # counter.update(text.lower().split(' '))
        items = items.replace('。', ',')
        tokens = list(jieba.cut(items))
        counter.update(tokens)
    words = [word for word, cnt in counter.items() if cnt > threshold and word != ',']
    print(words)
    vocab = Vocabulary()

    for word in words:
        print(word)
        vocab.add_word(word)
    return vocab


def main(json_file, threshold, vocab_path):
    vocab = build_vocab(json_file=json_file,
                        threshold=threshold)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size:{}".format(len(vocab)))
    print("Saved path in {}".format(vocab_path))


if __name__ == '__main__':
    main(json_file='../data/new_data/CN/Thyroid/captions.json',
         threshold=0,
         vocab_path='../data/new_data/CN/Thyroid/vocab.pkl')
