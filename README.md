# paddle_reformer
Reformer: The Efficient Transformer 论文复现 paddle2.x

# requirements
- transformers
- paddlenlp
- easydict
- torch
- paddle


# 目标
ReformerModel，ReformerForSequenceClassification和ReformerForQuestionAnswering网络前向推理输出对齐参考代码。
`注`: 由于`ReformerForSequenceClassification`和`ReformerForQuestionAnswering`都使用ReformerModel作为主干部分，因此只需要对齐ReformerModel部分前向传播的权重即可。

# （1）准备
- 从 https://huggingface.co/google/ 下载reformer权重pytorch_model.bin放入google下对应的文件夹
- 从 https://huggingface.co/junnyu/reformer_paddle 下载转化后的paddle权重放入paddle下对应的文件夹

# （2）eval模式对齐
**详细输出请查看日志：compare.log**
```python
# 在GPU和CPU模式下，进行loss和grad的比较
python compare_eval.py
```

# （3）train模式对齐（额外进行了train模式下，反向传播的梯度对齐）
```python
# 进入train文件夹
cd train
# 在GPU和CPU模式下，进行loss和grad的比较
python compare_train.py
# test on cpu!
# compare loss
# mean difference: tensor(8.5831e-06)
# max difference: tensor(8.5831e-06)
# ==================================================
# compare grad
# mean difference: tensor(1.6347e-11)
# max difference: tensor(1.4472e-08)
# ==================================================
# test on gpu!
# compare loss
# mean difference: tensor(4.7684e-07)
# max difference: tensor(4.7684e-07)
# ==================================================
# compare grad
# mean difference: tensor(8.7412e-11)
# max difference: tensor(4.4449e-08)
```

# （4）tokenizer对齐
```python
python compare_tokenizer.py 
['▁I', 't', '▁is', '▁a', '▁n', 'i', 'ce', '▁d', 'ay', '▁to', 'd', 'ay', '▁', ',', '▁I', '▁w', 'ant', '▁to', '▁go', '▁to', '▁the', '▁p', 'ar', 'k', '▁', '!']
['▁I', 't', '▁is', '▁a', '▁n', 'i', 'ce', '▁d', 'ay', '▁to', 'd', 'ay', '▁', ',', '▁I', '▁w', 'ant', '▁to', '▁go', '▁to', '▁the', '▁p', 'ar', 'k', '▁', '!']
==================================================
{'input_ids': [33, 260, 111, 4, 136, 264, 69, 30, 71, 26, 268, 71, 258, 277, 33, 8, 180, 26, 224, 26, 13, 40, 52, 282, 258, 287], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
{'input_ids': [33, 260, 111, 4, 136, 264, 69, 30, 71, 26, 268, 71, 258, 277, 33, 8, 180, 26, 224, 26, 13, 40, 52, 282, 258, 287], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

# 结语
改这个模型头都大了，其中api转换是大头！个人感觉paddle2.x的一些API好不人性化，比如gather，scatter,必须要用别的方法才能与pytorch的api对齐，之后有空再详细说下我是如何“曲线救国”的。
