import torch
import paddle
import json


from transformers import ReformerConfig
import paddle
from collections import OrderedDict

paddle.set_device("cpu")

def convert(infile,outfile,dtype="float64"):
    pt_model = torch.load(infile)
    d = OrderedDict()
    for k, v in pt_model.items():
        if "_mask_value_" in k:
            continue
        if v.ndim == 2 and "embedding" not in k:
            v = v.transpose(1, 0)
        d[k] = paddle.to_tensor(v.numpy().astype(dtype))

    paddle.save(d,outfile)


if __name__ == "__main__":
    convert("../google/reformer-enwik8/pytorch_model.bin","../paddle/reformer-enwik8/model_state.pdparams")