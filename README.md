# CNN-LSTM

## 论文

On the automatic generation of medical imaging reports (Coatt)

## 源码

https://github.com/ZexinYan/Medical-Report-Generation

# 数据集

- Liver_Dataset

- Mammary_Dataset

- Thyroid_Dataset

## 运行

- 运行utlis/prepare.py生成所需文件

- 运行ultis/bulid_vocab.py生成所需文件

- 修改路径，运行trainer

- 修改路径，运行tester

## 结果

结果存放在report_v4_models/v4/文件夹中，运行tester后会生成一个result的文件夹，其中存放了可视化的json和csv结果

## 备注

- 数据集的数据量除以batch_size的余数不能为1，否则会报错
- 在上述三个数据集跑会过拟合，源码在默认参数情况下在原数据集上跑也会过拟合或者不收敛

