# 在double精度下进行前向输出对齐

# 步骤
（1）准备
- 从 https://huggingface.co/google/ 下载reformer权重pytorch_model.bin放入google下对应的文件夹
- `python convert.py` 将权重转为paddle的类型（float64精度。）

（2）开始比较
- 在GPU和CPU模式下，对reformer-crime-and-punishment，进行loss和hidden states的比较
- 在GPU和CPU模式下，对reformer-enwik8，进行loss和hidden states的比较
- 其中`compare_long.py`走的逻辑和`compare_short.py`运行的逻辑有些许不同。（long的里面需要随机产生随机数，而short的不需要产生随机数）

# 结果

# 一：`python compare_long.py`
## cpu条件下，reformer-crime-and-punishment

```python
compare weights google/reformer-crime-and-punishment on cpu
Some weights of ReformerModelWithLMHead were not initialized from the model checkpoint at google/reformer-crime-and-punishment and are newly initialized: ['reformer.encoder.layers.4.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.3.attention.self_attention.self_mask_value_float16', 'reformer.encoder.layers.5.attention.self_attention.self_mask_value_float16', 'reformer.encoder.layers.0.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.2.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.2.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.5.attention.self_attention.self_mask_value_float32', 'reformer.encoder.layers.5.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.4.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.0.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.3.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.1.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.5.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.1.attention.self_attention.self_mask_value_float32', 'reformer.encoder.layers.3.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.3.attention.self_attention.self_mask_value_float32', 'reformer.encoder.layers.1.attention.self_attention.self_mask_value_float16', 'reformer.encoder.layers.1.attention.self_attention.mask_value_float32']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
compare loss
mean dif: 1.7763568394002505e-15
max dif: 1.7763568394002505e-15
compare hidden_states
mean dif: 0.0
max dif: 0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.480943742744044e-15
max dif: 3.907985046680551e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.0956069912607093e-15
max dif: 5.1514348342607263e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.233421388291687e-15
max dif: 6.750155989720952e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.843231131816731e-15
max dif: 9.059419880941277e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.930364053188986e-15
max dif: 1.4210854715202004e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 5.766000408711635e-15
max dif: 1.7763568394002505e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
==================================================
```

## gpu条件下，reformer-crime-and-punishment

```python
compare weights google/reformer-crime-and-punishment on gpu
W0908 10:11:16.075284  4857 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 8.6, Driver API Version: 11.4, Runtime API Version: 11.2
W0908 10:11:16.077921  4857 device_context.cc:422] device: 0, cuDNN Version: 8.1.
Some weights of ReformerModelWithLMHead were not initialized from the model checkpoint at google/reformer-crime-and-punishment and are newly initialized: ['reformer.encoder.layers.4.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.3.attention.self_attention.self_mask_value_float16', 'reformer.encoder.layers.5.attention.self_attention.self_mask_value_float16', 'reformer.encoder.layers.0.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.2.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.2.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.5.attention.self_attention.self_mask_value_float32', 'reformer.encoder.layers.5.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.4.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.0.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.3.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.1.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.5.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.1.attention.self_attention.self_mask_value_float32', 'reformer.encoder.layers.3.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.3.attention.self_attention.self_mask_value_float32', 'reformer.encoder.layers.1.attention.self_attention.self_mask_value_float16', 'reformer.encoder.layers.1.attention.self_attention.mask_value_float32']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
compare loss
mean dif: 0.0
max dif: 0.0
compare hidden_states
mean dif: 0.0
max dif: 0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.1703494052687356e-15
max dif: 2.3092638912203256e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.672947202537791e-15
max dif: 4.884981308350689e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.6500927583970937e-15
max dif: 5.684341886080802e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.171617291805587e-15
max dif: 6.394884621840902e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.13910961517481e-15
max dif: 1.1368683772161603e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.868770143296536e-15
max dif: 1.5631940186722204e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
****************************************************************************************************

```

## cpu条件下，reformer-enwik8

```python
compare weights google/reformer-enwik8 on cpu

compare loss
mean dif: 3.552713678800501e-15
max dif: 3.552713678800501e-15
compare hidden_states
mean dif: 0.0
max dif: 0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.7973976823188426e-15
max dif: 1.1723955140041653e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 7.717892643714277e-15
max dif: 3.979039320256561e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.0423239085105166e-14
max dif: 4.689582056016661e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.420261369343015e-14
max dif: 4.405364961712621e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.8846292068824215e-14
max dif: 5.115907697472721e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.4590748479072568e-14
max dif: 8.810729923425242e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.9682415723769335e-14
max dif: 9.094947017729282e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.522991425361118e-14
max dif: 1.0231815394945443e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.0819161928173984e-14
max dif: 1.0231815394945443e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.6629184490384635e-14
max dif: 1.4779288903810084e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 5.248073403562887e-14
max dif: 1.7053025658242404e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 6.128897424460583e-14
max dif: 2.2737367544323206e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
==================================================
```

## gpu条件下，reformer-enwik8

```python
compare weights google/reformer-enwik8 on gpu
compare loss
mean dif: 2.6645352591003757e-15
max dif: 2.6645352591003757e-15
compare hidden_states
mean dif: 0.0
max dif: 0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.757522007067951e-15
max dif: 6.394884621840902e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 5.2930474973557816e-15
max dif: 1.5631940186722204e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 7.342147429549854e-15
max dif: 1.8474111129762605e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.003657423320465e-14
max dif: 2.1316282072803006e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.3319471849094046e-14
max dif: 3.126388037344441e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.7260519763447273e-14
max dif: 4.831690603168681e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.1198124488869912e-14
max dif: 4.547473508864641e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.5395185493504112e-14
max dif: 5.400124791776761e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.967973792092364e-14
max dif: 5.684341886080801e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.400372917885404e-14
max dif: 9.094947017729282e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.877440517644211e-14
max dif: 1.2505552149377763e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.6007831807025245e-14
max dif: 1.8758328224066645e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```



# 二：`python compare_short.py`
## cpu条件下，reformer-crime-and-punishment

```python
compare weights google/reformer-crime-and-punishment on cpu
Some weights of ReformerModelWithLMHead were not initialized from the model checkpoint at google/reformer-crime-and-punishment and are newly initialized: ['reformer.encoder.layers.4.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.3.attention.self_attention.self_mask_value_float32', 'reformer.encoder.layers.1.attention.self_attention.self_mask_value_float16', 'reformer.encoder.layers.5.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.3.attention.self_attention.self_mask_value_float16', 'reformer.encoder.layers.1.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.4.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.1.attention.self_attention.self_mask_value_float32', 'reformer.encoder.layers.5.attention.self_attention.self_mask_value_float32', 'reformer.encoder.layers.0.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.3.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.1.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.5.attention.self_attention.self_mask_value_float16', 'reformer.encoder.layers.2.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.3.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.2.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.5.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.0.attention.self_attention.mask_value_float16']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
compare loss
mean dif: 1.7763568394002505e-15
max dif: 1.7763568394002505e-15
compare hidden_states
mean dif: 0.0
max dif: 0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.168750592530485e-15
max dif: 1.554312234475219e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.939823848536506e-15
max dif: 4.085620730620576e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.277647153952845e-15
max dif: 4.973799150320701e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.099114886618667e-15
max dif: 7.460698725481052e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 5.867734662380901e-15
max dif: 1.1368683772161603e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 6.89193839424817e-15
max dif: 1.5631940186722204e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
==================================================
```

## gpu条件下，reformer-crime-and-punishment

```python
compare weights google/reformer-crime-and-punishment on gpu
W0908 10:35:17.643958  1256 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 8.6, Driver API Version: 11.4, Runtime API Version: 11.2
W0908 10:35:17.646268  1256 device_context.cc:422] device: 0, cuDNN Version: 8.1.
Some weights of ReformerModelWithLMHead were not initialized from the model checkpoint at google/reformer-crime-and-punishment and are newly initialized: ['reformer.encoder.layers.4.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.3.attention.self_attention.self_mask_value_float32', 'reformer.encoder.layers.1.attention.self_attention.self_mask_value_float16', 'reformer.encoder.layers.5.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.3.attention.self_attention.self_mask_value_float16', 'reformer.encoder.layers.1.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.4.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.1.attention.self_attention.self_mask_value_float32', 'reformer.encoder.layers.5.attention.self_attention.self_mask_value_float32', 'reformer.encoder.layers.0.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.3.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.1.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.5.attention.self_attention.self_mask_value_float16', 'reformer.encoder.layers.2.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.3.attention.self_attention.mask_value_float32', 'reformer.encoder.layers.2.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.5.attention.self_attention.mask_value_float16', 'reformer.encoder.layers.0.attention.self_attention.mask_value_float16']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
compare loss
mean dif: 1.7763568394002505e-15
max dif: 1.7763568394002505e-15
compare hidden_states
mean dif: 0.0
max dif: 0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 9.119032037477873e-16
max dif: 1.4210854715202004e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.4534988230792656e-15
max dif: 2.1316282072803006e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.6312609461960298e-15
max dif: 4.263256414560601e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.3110238519646904e-15
max dif: 4.973799150320701e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.889358847413993e-15
max dif: 1.1368683772161603e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 5.768515750803484e-15
max dif: 2.0605739337042905e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
****************************************************************************************************

```

## cpu条件下，reformer-enwik8

```python
compare weights google/reformer-enwik8 on cpu
compare loss
mean dif: 3.552713678800501e-15
max dif: 3.552713678800501e-15
compare hidden_states
mean dif: 0.0
max dif: 0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.2503667780723344e-15
max dif: 1.3500311979441904e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.0493787307483535e-15
max dif: 4.547473508864641e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.3052253301224022e-14
max dif: 1.1368683772161603e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.533045968189174e-14
max dif: 1.2079226507921703e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.0518706750681618e-14
max dif: 1.1652900866465643e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.3893861346437284e-14
max dif: 1.4779288903810084e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.7444248761744714e-14
max dif: 1.4210854715202004e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.1032197621487565e-14
max dif: 1.4210854715202004e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.41099591347507e-14
max dif: 1.4210854715202004e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.0664766511208285e-14
max dif: 2.0463630789890885e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.346323071563602e-14
max dif: 2.2737367544323206e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 4.635937118612087e-14
max dif: 2.1032064978498966e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
==================================================
```

## gpu条件下，reformer-enwik8

```python
compare weights google/reformer-enwik8 on gpu
compare loss
mean dif: 3.552713678800501e-15
max dif: 3.552713678800501e-15
compare hidden_states
mean dif: 0.0
max dif: 0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 8.492301705718127e-16
max dif: 3.552713678800501e-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.691489078326759e-15
max dif: 1.2789769243681803e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 9.83141665696969e-15
max dif: 9.094947017729282e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.1631635460050388e-14
max dif: 9.094947017729282e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.55794947071309e-14
max dif: 8.526512829121202e-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 1.827758075598356e-14
max dif: 1.1937117960769683e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.0998322765653348e-14
max dif: 1.1652900866465643e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.404884944327496e-14
max dif: 1.2221335055073723e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 2.6661235186683714e-14
max dif: 1.2505552149377763e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.1754797048041684e-14
max dif: 1.5916157281026244e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.4256863407220135e-14
max dif: 1.5916157281026244e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean dif: 3.6835540488493614e-14
max dif: 1.6484591469634324e-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
