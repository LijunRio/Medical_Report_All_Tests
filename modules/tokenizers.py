import json
import re
from collections import Counter
from tqdm import tqdm
import jieba


class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path  # 报告路径
        self.threshold = args.threshold  # 词频下限
        self.dataset_name = args.dataset_name
        # if self.dataset_name == 'iu_xray':
        #     self.clean_report = self.clean_report_iu_xray
        # else:
        #     self.clean_report = self.clean_report_mimic_cxr  # 数据清洗函数
        self.clean_report = self.clean_report_ultrasound
        self.ann = json.loads(open(self.ann_path, 'r', encoding='utf-8-sig').read())  # 报告
        self.token2idx, self.idx2token = self.create_vocabulary()  # 构造词表

    def create_vocabulary(self):
        print('Buliding vocabulary...')
        total_tokens = []
        '''
        for example in tqdm(self.ann['test']):
            tokens = self.clean_report(example['report']).split()  # 将除句号的所有标点去掉，然后分词
            for token in tokens:
                total_tokens.append(token)
        '''
        for example in tqdm(self.ann['train']):
            tokens = jieba.cut(self.clean_report(example['finding'])) # 将除句号的所有标点去掉，然后分词
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)  # 计数
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']  # 保留高于词频阈值的词汇并加上<unk>
        vocab.sort()  # 按首字母排序
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):  # 构造idx2token和token2idx表
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        print('Building vocabulary complete!')
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_ultrasound(self, report):
        report_cleaner = lambda t: t.replace(',', '，').replace('-', '').replace('“', '').replace('”', '') \
            .replace('、', '，')
        report = report_cleaner(report)
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]  # 分句、转小写、把标点替换为空格
        report = ' . '.join(tokens) + ' .'  # 加句号
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = jieba.cut(self.clean_report(report))  # 清洗并分词
        ids = []
        for token in tokens:  # 词转idx
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]  # 开始与结束符
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
