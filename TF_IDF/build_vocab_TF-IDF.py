import pickle
from collections import Counter
import json
import re
import jieba


class JsonReader(object):
    def __init__(self, json_file):
        self.data = self.__read_json(json_file)
        self.keys = list(self.data.keys())

    def __read_json(self, filename):
        with open(filename, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        return data

    def __getitem__(self, item):
        # return self.data[item]
          return self.data[self.keys[item]]

    def __len__(self):
        return len(self.data)


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.id2word = {}
        self.idx = 0
        self.add_word('<pad>')  # 0
        self.add_word('<start>')  # 1
        self.add_word('<end>')  # 2
        self.add_word('<unk>')  # 3

    def add_word(self, word):
        if word not in self.word2idx:
            print(word)
            self.word2idx[word] = self.idx
            self.id2word[self.idx] = word
            self.idx += 1

    def get_word_by_id(self, id):
        # print(self.id2word[id])
        return self.id2word[id]

    def __call__(self, word):
        if word not in self.word2idx:
            # print(word)  # 句子没有分
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):  # 996
        # print(self.word2idx)  # 字典 word到id
        # print(self.id2word)  # 字典 id到word
        return len(self.word2idx)


def build_vocab(json_file, threshold):
    with open(json_file, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    caption_reader = data["train"]
    counter = Counter()

    for item in caption_reader:
        items = item['finding']

        '''
        report_cleaner = lambda t:t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(items) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        y = report.lower().split(' ')
        counter.update(y)
        '''

        tokens = list(jieba.cut(items))
        counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt > threshold and word != '']

    vocab = Vocabulary()
    json_vocab = []
    for word in words:
        json_vocab.append(word)
        vocab.add_word(word)
    length = len(json_vocab)
    print(length)

    #******************* selecting 800 words as the vocabulary for TF-IDF construction ***********************
    T = 799 - length
    words2 = [wor for wor, cnt in counter.items() if cnt == 3 and wor != '']
    for word in words2:
        if T > 0:
            json_vocab.append(word)
            vocab.add_word(word)
        T = T-1
    print(len(json_vocab))
    return vocab, json_vocab


def main(json_file, threshold, vocab_path, js):
    vocab, json_vocab = build_vocab(json_file=json_file, threshold=threshold)
    # js = '../data/CN/Mammary/vocab_TF-IDF.json'
    with open(js, 'w', encoding='utf-8-sig') as ff:
        json.dump(json_vocab, ff, ensure_ascii=False) # json序列化时默认使用过ascii编码，想要输出真正的中文需要指定ensure_ascii=False
    print("Total vocabulary size:{}".format(len(json_vocab)))

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Saved path in {}".format(vocab_path))


if __name__ == '__main__':
    main(json_file='../dataset/Ultrasonic_datasets/Liver_dataset/new_Liver2.json',
         threshold=3, vocab_path='../data/CN/Liver/vocab.pkl', js='../data/CN/Liver/vocab_TF-IDF.json')

    # main(json_file='/data/meihuan/data/ultrasound_data/Ultrasonic_datasets/Mammary_dataset/new_Mammary2.json',
    #      threshold=3, vocab_path='../data/CN/Mammary/vocab.pkl')
    # f = open('./vocab.json', 'r')
    # t = json.load(f)
    # t.sort()
