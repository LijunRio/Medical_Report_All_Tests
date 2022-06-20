# TriNet

## 论文

Joint Embedding of Deep Visual and Semantic Features for Medical Image Report Generation

## 源码链接

https://github.com/yangyan22/Medical-Report-Generation-TriNet

## 数据集

- Liver_Dataset
- Mammary_Dataset
- Thyroid_Dataset

## 运行

- 运行utils/bulid_vocab生成vocab.pkl
- 运行TF_IDF/bulid_vocab_TF-IDF生成vocab_TF-IDF.json，运行时回打印出词汇表长度，记录下来
- 运行TF_IDF/TF_IDF_MeRP生成TF_IDF_Report.json
- 修改utls/datasets中的__getitem__方法的DATA_PATH为图片位置
- 修改trainer的DATA_PATH、RESUME_MODEL_PATH、--model_path，将--report_dim改为词汇表长度，修改s_max（单篇报告最多句数），n_max（单句最多词数）为合适值
- 运行trainer
- 修改相关参数，运行tester

## 结果

--model_path中的print_epochs.csv保存的是pycocoevalcap的结果，report保存的是最后一个epoch的生成结果，--model_path/reuslt中保存的是验证集上得到最好结果的模型生成的报告

## 备注

- 数据集的数据量除以batch_size的余数不能为1，否则会报错

