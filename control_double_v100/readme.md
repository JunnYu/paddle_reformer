# 步骤
（0）环境：
- Python 3.8.10
- V100 32G
- 系统cuda 10.2 cudnn7.6
- paddle 2.1.3 cuda10.2   pytorch 1.9.1 cuda10.2

（1）准备
- 进入`cd weights/12_layer`
- `wget https://huggingface.co/google/reformer-enwik8/resolve/main/pytorch_model.bin`  #下载权重
- `cd ../../`返回上上级目录， `python convert.py` 将权重转为paddle的类型（float64精度。）

（2）比较:
- `python compare_long.py` 长的input_ids
- `python compare_short.py` 短的input_ids

（3）查看结果
- long_result.txt
- short_result.txt
