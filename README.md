# README



- 把超声数据集复制到data文件夹中

- 运行代码：以甲状腺为例

  ```
  python main.py
  --image_dir
  data/Ultrasonic_datasets/Throid_dataset/Thyroid_images
  --ann_path
  data/Ultrasonic_datasets/Throid_dataset/new_Thyroid2.json
  --dataset_name
  ultrasound
  --max_seq_length
  150
  --threshold
  3
  --batch_size
  16
  --epochs
  30
  --save_dir
  results/Thyroid
  --step_size
  1
  --gamma
  0.8
  --seed
  456789
  ```

- 代码运行起来比较慢，训练时间不长，但验证和测试都用了并行的beam search，导致推理时间很长。