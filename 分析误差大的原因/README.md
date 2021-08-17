# 结论
问题：精度误差为啥那么大？
答：paddle的框架问题！

```python
对于12层的模型。
使用预训练权重误差会达到（12_layer.txt文件最后一行）
mean dif: tensor(2.6582e-05) 
max dif: tensor(0.0009)
而使用随机初始化的权重，误差正常！（random_12_layer.txt文件最后一行）
mean dif: tensor(8.7896e-07) 
max dif: tensor(4.7684e-06)
```



# 分析过程

## 一：首先对比2_layer（这个是reformer-enwik8权重的前两层初始化的模型）和random_2_layer（这个是随机初始化的模型。）
他们两个的模型结果如下：
`"attn_layers": ["local","local"]`使用的是2个local attetnion层。


（1）查看2_layer模型内部结果精度对齐情况：

这里转换后的2层模型请从网盘下载：然后放进`weights/2_layer`文件夹！
链接：https://pan.baidu.com/s/1RGLTnImt5sPQYy-Y1u6XRA 
提取码：wbm4

- 取消compare_eval.py中的注释test_cpu("weights/2_layer")。
- 运行`python compare_eval.py`。
- 查看`weights/2_layer.txt`中的结果。

下面阐述该模型内部结果精度对齐的问题！
```
第一层local attention！
==========================================
0 （这里的id对应于reformer/modeling.py-L1664）
这比较了embedding层的输出结果，可以发现结果一致。
mean dif: tensor(0.) 
max dif: tensor(0.)
==========================================
1 （这里的id对应于reformer/modeling.py-L1669）
这里是通过一个layer_norm后的结果比较。(为之后误差累计埋下伏笔，问题：相同的输出进去，为什么layer norm会有误差？)
mean dif: tensor(2.4824e-08) 
max dif: tensor(4.1723e-07)
==========================================
2（这里的id对应于reformer/modeling.py-L1376）
这里是通过query隐射的全连接后的结果，可以发现最大误差竟然从10-7变成了10-6。
mean dif: tensor(2.1102e-07) 
max dif: tensor(3.0994e-06)
==========================================
3（这里的id对应于reformer/modeling.py-L1378）
这里是通过key隐射的全连接后的结果，可以发现最大误差竟然从10-7变成了10-6。
mean dif: tensor(2.1887e-07) 
max dif: tensor(3.5763e-06)
==========================================
4（这里的id对应于reformer/modeling.py-L1380）
这里是通过value隐射的全连接后的结果，可以发现最大误差竟然从10-7变成了10-6。
mean dif: tensor(1.5330e-07) 
max dif: tensor(2.6226e-06)
==========================================
5（这里的id对应于reformer/modeling.py-L1393）
这里是对query进行了拆分操作，拆分出attention head，误差和之前的结果一致。
mean dif: tensor(2.1102e-07) 
max dif: tensor(3.0994e-06)
==========================================
6（这里的id对应于reformer/modeling.py-L1395）
这里是对key进行了拆分操作，拆分出attention head，误差和之前的结果一致。
mean dif: tensor(2.1887e-07) 
max dif: tensor(3.5763e-06)
==========================================
7（这里的id对应于reformer/modeling.py-L1397）
这里是对value进行了拆分操作，拆分出attention head，误差和之前的结果一致。
mean dif: tensor(1.5330e-07) 
max dif: tensor(2.6226e-06)
==========================================
8（这里的id对应于reformer/modeling.py-L1418）
这里进行了如下的操作，主要是对key_vectors进行缩放！
key_vectors = key_vectors / paddle.sqrt(paddle.to_tensor(self.attention_head_size, dtype=key_vectors.dtype))
mean dif: tensor(1.9347e-08) 
max dif: tensor(3.2783e-07)
==========================================
9（这里的id对应于reformer/modeling.py-L1425）
这里进行了如下的操作，误差当然为0！
indices = paddle.tile(
    paddle.arange(sequence_length),
    repeat_times=[batch_size, self.num_attention_heads, 1],
)
mean dif: tensor(0.) 
max dif: tensor(0.)
==========================================
10（这里的id对应于reformer/modeling.py-L1481）
这里进行了如下操作，主要是query_vectors和key_vectors进行矩阵乘法！可以发现最大误差变成了1.2398e-05？？？？？？？？
query_key_dots = paddle.matmul(query_vectors, key_vectors, transpose_y=True)
mean dif: tensor(6.7489e-07) 
max dif: tensor(1.2398e-05)
==========================================
11（这里的id对应于reformer/modeling.py-L1493）
这里是计算mask的。结果一致！
mean dif: tensor(0.) 
max dif: tensor(0.)
==========================================
12（这里的id对应于reformer/modeling.py-L1505）
这里是对query_key_dots施加mask操作的，可以发现最大误差还是一样！还是1.2398e-05？？？？？？？？？？？？？
query_key_dots = paddle.where(mask.astype(paddle.bool), query_key_dots, mask_value)
mean dif: tensor(3.5710e-07) 
max dif: tensor(1.2398e-05)
==========================================
13（这里的id对应于reformer/modeling.py-L1513）
这里进行了如下操作，主要进行了logsumexp的操作，最大误差固定在1.2398e-05。
logits = _logsumexp(query_key_dots, axis=-1, keepdim=True)
mean dif: tensor(1.7348e-06) 
max dif: tensor(1.2398e-05)
==========================================
14（这里的id对应于reformer/modeling.py-L1516）
这里进行了如下操作，进行了个减法，然后通过exp，最大误差1.4305e-06。
attention_probs = paddle.exp(query_key_dots - logits)
mean dif: tensor(1.3063e-08) 
max dif: tensor(1.4305e-06)
==========================================
15（这里的id对应于reformer/modeling.py-L1528）
这里进行了如下操作，最大误差6.1393e-06
out_vectors = paddle.matmul(attention_probs, value_vectors)
mean dif: tensor(2.3706e-07) 
max dif: tensor(6.1393e-06)
==========================================
16（这里的id对应于reformer/modeling.py-L1546）
这里进行了_merge_hidden_size_dims操作，进行的是维度合并，误差不变。
mean dif: tensor(2.3706e-07) 
max dif: tensor(6.1393e-06)
==========================================
17（这里的id对应于reformer/modeling.py-L1616）
这里的ReformerSelfOutput的输入部分，也就上面16的输出。结果一模一样的。
mean dif: tensor(2.3706e-07) 
max dif: tensor(6.1393e-06)
==========================================
18（这里的id对应于reformer/modeling.py-L1619）
上面的结果随后输入到一个全连接层！最大误差变成了1.2398e-05？？？？？？？？？？？？
self.dense(hidden_states)
mean dif: tensor(1.1895e-06) 
max dif: tensor(1.2398e-05)
==========================================
19（这里的id对应于reformer/modeling.py-L1881）
这里误差为0，由于是第一层的原因，他的输入prev_attn_output都是embedding的输出部分。（第一层的prev_attn_output和hidden_states都是一样的）
mean dif: tensor(0.) 
max dif: tensor(0.)
==========================================
20（这里的id对应于reformer/modeling.py-L1884）
然后将prev_attn_output和attention部分的输出相加，发现最大误差为1.2398e-05。
attn_output = prev_attn_output + attn_output
mean dif: tensor(1.1897e-06) 
max dif: tensor(1.2398e-05)
==========================================
21（这里的id对应于reformer/modeling.py-L1895）
这里误差为0，由于是第一层的原因，他的输入hidden_states都是embedding的输出部分。（第一层的prev_attn_output和hidden_states都是一样的）
mean dif: tensor(0.) 
max dif: tensor(0.)
==========================================
22（这里的id对应于reformer/modeling.py-L1785）
这里对attn_output进行了layer norm，发现最大误差1.2219e-06。
mean dif: tensor(1.0983e-07) 
max dif: tensor(1.2219e-06)
==========================================
23（这里的id对应于reformer/modeling.py-L1788）
这里进行ReformerFeedForwardDense操作，进行了全连接和relu操作。最大误差8.9407e-06，快要进入10-5级别的苗头了。
mean dif: tensor(3.0540e-08) 
max dif: tensor(8.9407e-06)
==========================================
24（这里的id对应于reformer/modeling.py-L1756）
这里进行了ReformerFeedForwardOutput操作，进行了全连接。可以发现最大误差果然变大了，成了4.1962e-05。
mean dif: tensor(1.6455e-06) 
max dif: tensor(4.1962e-05)
==========================================
25（这里的id对应于reformer/modeling.py-L1898）
这里进行了两个结果的相加，hidden_states的误差为0，self.feed_forward(attn_output)的误差4.1962e-05。
hidden_states = hidden_states + self.feed_forward(attn_output)
mean dif: tensor(1.6459e-06) 
max dif: tensor(4.1962e-05)
==========================================

总结一下，通过一层的local attention,最大误差成了4.1962e-05？？？？，误差这么大？？？？？随后在这个基础上继续进行计算，误差会不断累积。


########################################################################################
第二层local attention！模型结果一样，这里的26与上面的0相对应，同理27与1对应。
这部分的介绍会稍许简略。
==========================================
26
这里对应于第一层最后的输出。
mean dif: tensor(1.6459e-06) 
max dif: tensor(4.1962e-05)
==========================================
27
然后layer norm，稍微把最大误差降低了点。
mean dif: tensor(1.1579e-07) 
max dif: tensor(1.1921e-06)
==========================================
28
query隐射
mean dif: tensor(6.1130e-07) 
max dif: tensor(6.4969e-06)
==========================================
29
key隐射，可以发现这个key隐射后的结果，最大误差1.0550e-05。
mean dif: tensor(6.5241e-07) 
max dif: tensor(1.0550e-05)
==========================================
30
value隐射
mean dif: tensor(5.5772e-07) 
max dif: tensor(4.8876e-06)
==========================================
31
query拆分attetnion head出来
mean dif: tensor(6.1130e-07) 
max dif: tensor(6.4969e-06)
==========================================
32
key拆分attetnion head出来
mean dif: tensor(6.5241e-07) 
max dif: tensor(1.0550e-05)
==========================================
33
value拆分attetnion head出来
mean dif: tensor(5.5772e-07) 
max dif: tensor(4.8876e-06)
==========================================
34
对key_vectors进行缩放！
mean dif: tensor(5.7687e-08) 
max dif: tensor(9.3132e-07)
==========================================
35
计算的indices
mean dif: tensor(0.) 
max dif: tensor(0.)
==========================================
36
这里进行矩阵乘法，
mean dif: tensor(6.1130e-07) 
max dif: tensor(6.4969e-06)
与
mean dif: tensor(5.7687e-08) 
max dif: tensor(9.3132e-07)
这两个矩阵乘法后的最大误差结果为2.0981e-05，可以说是很大了。

paddle.matmul(query_vectors, key_vectors, transpose_y=True)
mean dif: tensor(1.6888e-06) 
max dif: tensor(2.0981e-05)
==========================================
37
计算mask
mean dif: tensor(0.) 
max dif: tensor(0.)
==========================================
38
对query_key_dots施加mask。
mean dif: tensor(8.7723e-07) 
max dif: tensor(2.0981e-05)
==========================================
39
通过logsumexp
mean dif: tensor(2.1962e-06) 
max dif: tensor(2.0981e-05)
==========================================
40
进行paddle.exp(query_key_dots - logits)
mean dif: tensor(2.6368e-08) 
max dif: tensor(2.3842e-06)
==========================================
41
进行paddle.matmul(attention_probs, value_vectors)，看最大误差还行8.7023e-06，不过有接近10-5的苗头，一层全连接就能把他拉近10-5。
mean dif: tensor(5.6485e-07) 
max dif: tensor(8.7023e-06)
==========================================
42
合并attention head操作。
mean dif: tensor(5.6485e-07) 
max dif: tensor(8.7023e-06)
==========================================
43
这里的ReformerSelfOutput的输入部分，也就上面42的输出。结果一模一样的。
mean dif: tensor(5.6485e-07) 
max dif: tensor(8.7023e-06)
==========================================
44
再来一层全连接操作，误差从10-6变成了10-5。
mean dif: tensor(2.2194e-06) 
max dif: tensor(1.5736e-05)
==========================================
45
这里保存的是prev_attn_output，我们目前处于第二层，这个是第一层的attn部分的output。可以查看第一层的20那部分。
mean dif: tensor(1.1897e-06) 
max dif: tensor(1.2398e-05)
==========================================
46
如下所示，这里是相加后的误差，
attn_output = prev_attn_output + attn_output
mean dif: tensor(2.5309e-06) 
max dif: tensor(1.8597e-05)
==========================================
47
这里是第二层的输入hidden_states，也就是第一层最后的输出，可以查看第一层的25那部分。
mean dif: tensor(1.6459e-06) 
max dif: tensor(4.1962e-05)
==========================================
48
这里对attn_output进行了layer norm，发现最大误差2.0266e-06。
mean dif: tensor(2.5753e-07) 
max dif: tensor(2.0266e-06)
==========================================
49
这里进行ReformerFeedForwardDense操作，进行了全连接和relu操作。最大误差1.3888e-05了。
mean dif: tensor(5.7802e-08) 
max dif: tensor(1.3888e-05)
==========================================
50
这里进行了ReformerFeedForwardOutput操作，进行了全连接。可以发现最大误差果然变大了，由1.3888e-05变成0.0001？？？？直接迈入10-4级别了！！！！
mean dif: tensor(2.6400e-06) 
max dif: tensor(0.0001)
==========================================
51
这里进行了两个结果的相加。hidden_states = hidden_states + self.feed_forward(attn_output)
hidden_states的误差
mean dif: tensor(1.6459e-06) 
max dif: tensor(4.1962e-05)
self.feed_forward(attn_output)的误差
mean dif: tensor(2.6400e-06) 
max dif: tensor(0.0001)
这两个相加后最大误差0.0001，10-4级别。
mean dif: tensor(3.2313e-06) 
max dif: tensor(0.0001)
```

总结：这里只有两层,local attention，没想到最后最大的误差到了10-4级别，如果后面再加好多好多层，其最终的结果可想而知，最终会到达10-3的误差！！！！！！





（2）查看random_2_layer模型内部结果精度对齐情况：（这个模型结构和上述一样，只不过权重是随机初始化的！）
- `python generate.py`生成随机初始化的2层的模型！！！
- 取消compare_eval.py中的注释test_cpu("weights/random_2_layer")。
- 运行`python compare_eval.py`。
- 查看`weights/random_2_layer.txt`中的结果。


```
==========================================
0
mean dif: tensor(0.) 
max dif: tensor(0.)
==========================================
1
mean dif: tensor(1.2322e-07) 
max dif: tensor(1.6689e-06)
==========================================
2
mean dif: tensor(1.3766e-07) 
max dif: tensor(1.4305e-06)
==========================================
3
mean dif: tensor(1.3902e-07) 
max dif: tensor(1.6689e-06)
==========================================
4
mean dif: tensor(1.3823e-07) 
max dif: tensor(1.6689e-06)
==========================================
5
mean dif: tensor(1.3766e-07) 
max dif: tensor(1.4305e-06)
==========================================
6
mean dif: tensor(1.3902e-07) 
max dif: tensor(1.6689e-06)
==========================================
7
mean dif: tensor(1.3823e-07) 
max dif: tensor(1.6689e-06)
==========================================
8
mean dif: tensor(1.2292e-08) 
max dif: tensor(1.4901e-07)
==========================================
9
mean dif: tensor(0.) 
max dif: tensor(0.)
==========================================
10
mean dif: tensor(1.5376e-07) 
max dif: tensor(1.4305e-06)
==========================================
11
mean dif: tensor(0.) 
max dif: tensor(0.)
==========================================
12
mean dif: tensor(7.9819e-08) 
max dif: tensor(1.1325e-06)
==========================================
13
mean dif: tensor(1.1972e-07) 
max dif: tensor(9.5367e-07)
==========================================
14
mean dif: tensor(5.5628e-09) 
max dif: tensor(2.9802e-07)
==========================================
15
mean dif: tensor(7.0004e-08) 
max dif: tensor(9.5367e-07)
==========================================
16
mean dif: tensor(7.0004e-08) 
max dif: tensor(9.5367e-07)
==========================================
17
mean dif: tensor(7.0004e-08) 
max dif: tensor(9.5367e-07)
==========================================
18
mean dif: tensor(6.8928e-08) 
max dif: tensor(8.3447e-07)
==========================================
19
mean dif: tensor(0.) 
max dif: tensor(0.)
==========================================
20
mean dif: tensor(6.9088e-08) 
max dif: tensor(8.3447e-07)
==========================================
21
mean dif: tensor(0.) 
max dif: tensor(0.)
==========================================
22
mean dif: tensor(1.6649e-07) 
max dif: tensor(1.9073e-06)
==========================================
23
mean dif: tensor(8.4228e-08) 
max dif: tensor(1.6689e-06)
==========================================
24
mean dif: tensor(1.9400e-07) 
max dif: tensor(1.5497e-06)
==========================================
25
mean dif: tensor(1.9382e-07) 
max dif: tensor(1.5497e-06)
==========================================
26
mean dif: tensor(1.9382e-07) 
max dif: tensor(1.5497e-06)
==========================================
27
mean dif: tensor(2.4730e-07) 
max dif: tensor(3.0994e-06)
==========================================
28
mean dif: tensor(2.1674e-07) 
max dif: tensor(1.6689e-06)
==========================================
29
mean dif: tensor(2.1619e-07) 
max dif: tensor(1.6689e-06)
==========================================
30
mean dif: tensor(2.1606e-07) 
max dif: tensor(1.9073e-06)
==========================================
31
mean dif: tensor(2.1674e-07) 
max dif: tensor(1.6689e-06)
==========================================
32
mean dif: tensor(2.1619e-07) 
max dif: tensor(1.6689e-06)
==========================================
33
mean dif: tensor(2.1606e-07) 
max dif: tensor(1.9073e-06)
==========================================
34
mean dif: tensor(1.9104e-08) 
max dif: tensor(1.4901e-07)
==========================================
35
mean dif: tensor(0.) 
max dif: tensor(0.)
==========================================
36
mean dif: tensor(2.1637e-07) 
max dif: tensor(1.6689e-06)
==========================================
37
mean dif: tensor(0.) 
max dif: tensor(0.)
==========================================
38
mean dif: tensor(1.0991e-07) 
max dif: tensor(1.4305e-06)
==========================================
39
mean dif: tensor(1.4162e-07) 
max dif: tensor(8.6427e-07)
==========================================
40
mean dif: tensor(6.9334e-09) 
max dif: tensor(2.6822e-07)
==========================================
41
mean dif: tensor(1.0241e-07) 
max dif: tensor(1.9073e-06)
==========================================
42
mean dif: tensor(1.0241e-07) 
max dif: tensor(1.9073e-06)
==========================================
43
mean dif: tensor(1.0241e-07) 
max dif: tensor(1.9073e-06)
==========================================
44
mean dif: tensor(9.0156e-08) 
max dif: tensor(1.3113e-06)
==========================================
45
mean dif: tensor(6.9088e-08) 
max dif: tensor(8.3447e-07)
==========================================
46
mean dif: tensor(1.1860e-07) 
max dif: tensor(1.5497e-06)
==========================================
47
mean dif: tensor(1.9382e-07) 
max dif: tensor(1.5497e-06)
==========================================
48
mean dif: tensor(2.1504e-07) 
max dif: tensor(2.1458e-06)
==========================================
49
mean dif: tensor(9.7017e-08) 
max dif: tensor(2.1458e-06)
==========================================
50
mean dif: tensor(2.1296e-07) 
max dif: tensor(1.6689e-06)
==========================================
51
mean dif: tensor(2.9304e-07) 
max dif: tensor(2.1458e-06)
```

结果很正常！**说明我搭建的模型结果没有问题**！！，可以尝试使用generate.py生成不同初始化的模型，然后比较，最终发现结果还是正确的！！！！！！！


## 二：再来对比12_layer（这个是用reformer-enwik8权重初始化的模型）和random_12_layer（这个是随机初始化的模型。）

（1）查看12_layer模型内部结果精度对齐情况：（这里由于使用到了预训练权重，因此需要下载预训练权重然后放进`weights/12_layer`文件夹）

paddle版本  https://huggingface.co/junnyu/reformer_paddle/tree/main/reformer-enwik8
huggingface版本 https://huggingface.co/google/reformer-enwik8/tree/main


- 取消compare_eval.py中的注释test_cpu("weights/12_layer")。
- 运行`python compare_eval.py`。
- 查看`weights/12_layer.txt`中的结果。

由于这个有12层，行数太多，请自行进入`weights/12_layer.txt`查看结果，分析同上！最终模型最大误差max dif: tensor(0.0009)！！！！！！！



（2）查看random_12_layer模型内部结果精度对齐情况：
- `python generate.py`生成随机初始化的12层的模型！！！
- 取消compare_eval.py中的注释test_cpu("weights/random_12_layer")。
- 运行`python compare_eval.py`。
- 查看`weights/random_12_layer.txt`中的结果。

由于这个有12层，行数太多，请自行进入`weights/random_12_layer.txt`查看结果，分析同上！
最终模型最大误差max dif: tensor(4.7684e-06)！！！！！！！

```python
那么问题来了，为什么随机初始化的跟加载预训练权重的计算出来的结果误差那么大？
（注：我确保模型权重转化是正确的,不然平均误差不会这么点！）
可能原因：
-（1）框架问题！！！！
-（2）框架问题！！！！
-（3）框架问题！！！！
```
