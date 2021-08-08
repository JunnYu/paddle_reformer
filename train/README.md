## 比较训练过程的梯度

为了进行train模式下的精度对齐，我进行了如下修改。
### （1）修改1-固定随机数
reformer/modeling.py Line789部分原本是使用`paddle.randn`生成的随机数为了进行模型对齐，我这边从本地加载同样的`random_rotations`以求最终的结果对齐。
```python
        # TODO
        # create a random self.attention_head_size x num_hashes x num_buckets/2
        # random_rotations = paddle.randn(shape=rotations_shape, dtype=vectors.dtype)
        random_rotations = paddle.load("buckets/bucks.pd")
        # Output dim: Batch_Size x Num_Attn_Heads x Num_Hashes x Seq_Len x Num_Buckets/2
        rotated_vectors = einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
```
transformers/models/reformer/modeling_reformer.py Line734部分原本是使用`torch.randn`生成的随机数为了进行模型对齐，我这边从本地加载同样的`random_rotations`以求最终的结果对齐。
```python
        # TODO
        # random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)
        random_rotations = torch.load("buckets/bucks.pt").to(vectors.device)
        # Output dim: Batch_Size x Num_Attn_Heads x Num_Hashes x Seq_Len x Num_Buckets/2
        rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)
```
### （2）修改2-将dropout置为0，为了消除dropout随机带来的影响
`reformer-crime-and-punishment-64-128/config.json`和`reformer-crime-and-punishment-64-128/model_config.json`中有关`dropout`的我都置为了0！

### （3）缩小`axial_pos_shape`
`reformer-crime-and-punishment-64-128/config.json`和`reformer-crime-and-punishment-64-128/model_config.json`原版的`axial_pos_shape`是`[512,1024]`,
这么说训练的时候我的输入必须是`[bs,512*1024]`这个形状，这个tensor太长了，我的6G显卡OOM了。为了方便调试，我将其修改成了`[64,128]`。


# Tips:
train模式下，前向传播和反向传播过程的结果都是在上述基础上进行的。