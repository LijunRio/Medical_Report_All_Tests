import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import numpy as np
from torchvision import transforms
import pickle
import matplotlib.pyplot as plt
import jieba


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

            self.word2idx[word] = self.idx
            self.id2word[self.idx] = word
            self.idx += 1

    def get_word_by_id(self, id):
        # print(self.id2word[id])
        return self.id2word[id]

    # 如果word不在词汇表中，返回unk，否则返回idx有
    def __call__(self, word):
        if word not in self.word2idx:
            # print(word)
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        # print(self.word2idx)
        # print(self.id2word)
        return len(self.word2idx)


class ChestXrayDataSet(Dataset):
    def __init__(self,
                 data_dir,
                 split,
                 vocabulary,
                 Report_path,
                 transforms,
                 DATA_PATH,
                 s_max=7,
                 n_max=20):

        self.vocab = vocabulary
        self.transform = transforms
        self.s_max = s_max # 一个report的最大句子数
        self.n_max = n_max # 单句最大词汇数
        self.image1, self.image2, self.caption, self.report = self.__load_label_list(data_dir, split, Report_path)
        self.DATA_PATH = DATA_PATH

    def __load_label_list(self, data_dir, split, Report_path):
        with open(data_dir, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        data_all = data[split]

        with open(Report_path, 'r') as f:
            R = json.load(f)

        image1 = []
        image2 = []
        labels = []
        report = []
        if split == "train":
            for line in range(len(data_all)):
                image_name1 = data_all[line]['image_path'][0]
                image1.append(image_name1)
                image_name2 = data_all[line]['image_path'][1]
                image2.append(image_name2)
                labels.append(data_all[line]['finding'])
                report.append(R[str(data_all[line]['uid'])])

            return image1, image2, labels, report

        if split == "test":
            for line in range(len(data_all)):
                image_name1 = data_all[line]['image_path'][0]
                image1.append(image_name1)
                image_name2 = data_all[line]['image_path'][1]
                image2.append(image_name2)
                labels.append(data_all[line]['finding'])
                report.append(0)
            return image1, image2, labels, report

        if split == "val":
            for line in range(len(data_all)):
                image_name1 = data_all[line]['image_path'][0]
                image1.append(image_name1)
                image_name2 = data_all[line]['image_path'][1]
                image2.append(image_name2)
                labels.append(data_all[line]['finding'])
                report.append(0)
            return image1, image2, labels, report

    def __getitem__(self, index):
        image_1 = self.image1[index] 
        image_2 = self.image2[index]

        # DATA_PATH = "D:/RIO/All_Datastes/整理好的超声数据集/Ultrasonic_datasets/Throid_dataset/Thyroid_images/"
        # DATA_PATH = "/data/meihuan/data/ultrasound_data/Ultrasonic_datasets/Throid_dataset/Thyroid_images/"
        # DATA_PATH = "/data/meihuan/data/ultrasound_data/Ultrasonic_datasets/Mammary_dataset/Mammary_images/"
        id = image_1.split("_")[0]  # 去掉后缀作为id

        image1 = Image.open(''.join([self.DATA_PATH, image_1])).convert('RGB')
        image2 = Image.open(''.join([self.DATA_PATH, image_2])).convert('RGB')

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

            text = self.caption[index] # report
            # text = text.lower()
            # text = text.replace(',', '')
        target = list()
        max_word_num = 0
        text = text.replace('。', '，')
        text = text.replace(',', '，')
        text = text.split('，')
        for i, sentence in enumerate(text):
        # for i, sentence in enumerate(text.replace('。', ',').split(',')):
            if i >= self.s_max:
                break # 跳过该report
            # sentence = sentence.replace('.', '')
            # sentence = sentence.split() # 单句分词
            sentence = list(jieba.cut(sentence))
            if len(sentence) == 0 or len(sentence) == 1 or len(sentence) > self.n_max:
                continue # 跳过该句
            
            # 每个句子组装成一个list
            tokens = list()
            tokens.append(self.vocab('<start>'))
            tokens.extend([self.vocab(token) for token in sentence])
            tokens.append(self.vocab('<end>'))


            if max_word_num < len(tokens): # 单句最大词汇数
                max_word_num = len(tokens)
            target.append(tokens) # 句子list组装成一个list
        sentence_num = len(target) # 一篇report的句子数
        report = self.report[index] # TF-IDF report
        return image1, image2, target, report, sentence_num, max_word_num, id

    def __len__(self):
        return len(self.image1)


def collate_fn(data):   
    image1, image2, captions, report, sentence_num, max_word_num, id = zip(*data)

    images1 = torch.stack(image1, 0)  
    images2 = torch.stack(image2, 0)

    max_sentence_num = max(sentence_num)
    max_word_num = max(max_word_num)

    targets = np.zeros((len(captions), max_sentence_num + 1, max_word_num))
    prob = np.zeros((len(captions), max_sentence_num + 1))

    for i, caption in enumerate(captions):
        for j, sentence in enumerate(caption):
            targets[i, j, :len(sentence)] = sentence[:]
            prob[i][j] = len(sentence) > 0

    return images1, images2, targets, prob, report, id


def get_loader(data_dir,
               split,
               vocabulary,
               Report_path,
               transform,
               batch_size,
               image_pth,
               s_max,
               n_max,
               shuffle=False):

    dataset = ChestXrayDataSet(data_dir=data_dir,
                               split=split,
                               vocabulary=vocabulary,
                               Report_path=Report_path,
                               transforms=transform,
                               DATA_PATH = image_pth,
                               s_max=s_max,
                               n_max=n_max)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader


# if __name__ == '__main__':
#     DATA_PATH = 'D:/postgraduate0/Medical-Report-Generation-TriNet/Medical-Report-Generation-TriNet-main/data/CN/Mammary'
#     vocab_path = DATA_PATH + '/vocab.pkl'
#     data_path = DATA_PATH + '/Mammary_annotation.json'
#     split = 'train'
#     Report_path = DATA_PATH + '/TF_IDF_Report.json'
#
#     with open(vocab_path, 'rb') as f:
#         vocab = pickle.load(f)
#     # print("Vocab Size:{}\n".format(len(vocab)))
#
#     batch_size = 2
#     resize = 224
#     crop_size = 224
#     transform = transforms.Compose([
#         transforms.Resize((resize, resize)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406),
#                              (0.229, 0.224, 0.225))])
#
#     data_loader = get_loader(data_dir=data_path,
#                              split=split,
#                              vocabulary=vocab,
#                              Report_path=Report_path,
#                              transform=transform,
#                              batch_size=batch_size,
#                              image_pth = imagePth,
#                              s_max=25,
#                              n_max=10,
#                              shuffle=True)
#
#     for i, (images1, images2, targets, prob, report, id) in enumerate(data_loader):
#         # print(images1.shape)  # torch.Size([BS, 3, 224, 224])
#         # print(images2.shape)  # torch.Size([BS, 3, 224, 224])
#         plt.imshow(images1[0][0])
#         plt.show()
#         plt.imshow(images2[0][0])
#         plt.show()
#         plt.imshow(images1[1][0])
#         plt.show()
#         plt.imshow(images2[1][0])
#         plt.show()
#         print(targets)
#         print(prob)
#         print(id)
#         print(report)
#         break
