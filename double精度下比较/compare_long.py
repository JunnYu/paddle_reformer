import torch
from transformers.models.reformer.modeling_reformer import (
    ReformerModelWithLMHead as PTReformerModelWithLMHead,
)
import numpy as np
import paddle
from reformer import ReformerModelWithLMHead as PDReformerModelWithLMHead
import shutil
import os
paddle.set_grad_enabled(False)
torch.set_grad_enabled(False)


def create_inputs(device="cpu"):
    paddle.seed(42)
    pdx = {
        "input_ids": paddle.randint(50, 150, shape=[2, 1024]),
        "attention_mask": paddle.randint(1, 2, shape=[2, 1024]),
    }
    ptx = {k:torch.from_numpy(v.numpy()).to(device) for k,v in pdx.items()}

    
    return pdx,ptx


def compare(a, b):
    a = a.numpy()
    b = b.numpy()
    print("mean dif:", np.abs(a - b).mean())
    print("max dif:", np.abs(a - b).max())


def test_cpu(model_path):
    try:
        shutil.rmtree("buckets")
    except:
        pass
    os.makedirs("buckets")
    print(f"compare weights {model_path} on cpu")
    paddle.set_device("cpu")
    paddle.set_default_dtype("float64")
    
    pd_model = PDReformerModelWithLMHead.from_pretrained(
        model_path.replace("google", "paddle")
    )
    
    pd_model.eval()
    
    
    pt_model = PTReformerModelWithLMHead.from_pretrained(model_path).double()
    pt_model.eval()
    pd_inputs, pt_inputs = create_inputs(device="cpu")
    pd_outputs = pd_model(
        pd_inputs["input_ids"],
        attention_mask=pd_inputs["attention_mask"],
        labels=pd_inputs["input_ids"],
        output_hidden_states=True,
    )
    pt_outputs = pt_model(
        pt_inputs["input_ids"],
        attention_mask=pt_inputs["attention_mask"],
        labels=pt_inputs["input_ids"],
        output_hidden_states=True,
    )
    print("compare loss")
    compare(pd_outputs.loss, pt_outputs.loss)
    print("compare hidden_states")
    for a, b in zip(pd_outputs.hidden_states, pt_outputs.hidden_states):
        compare(a, b)
        print("~" * 50)


def test_gpu(model_path):
    try:
        shutil.rmtree("buckets")
    except:
        pass
    os.makedirs("buckets")
    paddle.set_device("gpu")
    paddle.set_default_dtype("float64")
    print(f"compare weights {model_path} on gpu")
    pd_model = PDReformerModelWithLMHead.from_pretrained(
        model_path.replace("google", "paddle")
    )
    pd_model.eval()
    pt_model = PTReformerModelWithLMHead.from_pretrained(model_path).double()
    pt_model.to("cuda:0")
    pt_model.eval()
    pd_inputs, pt_inputs = create_inputs(device="cuda:0")
    pd_outputs = pd_model(
        pd_inputs["input_ids"],
        attention_mask=pd_inputs["attention_mask"],
        labels=pd_inputs["input_ids"],
        output_hidden_states=True,
    )
    pt_outputs = pt_model(
        pt_inputs["input_ids"],
        attention_mask=pt_inputs["attention_mask"],
        labels=pt_inputs["input_ids"],
        output_hidden_states=True,
    )
    print("compare loss")
    compare(pd_outputs.loss, pt_outputs.loss.cpu())
    print("compare hidden_states")
    for a, b in zip(pd_outputs.hidden_states, pt_outputs.hidden_states):
        compare(a, b.cpu())
        print("~" * 50)


if __name__ == "__main__":
    # 比较reformer-crime-and-punishment

    test_cpu("google/reformer-crime-and-punishment")
    print("=" * 50)

    test_gpu("google/reformer-crime-and-punishment")
    print("*" * 100)
    
    # 比较reformer-enwik8
    test_cpu("google/reformer-enwik8")
    print("=" * 50)
    

    test_gpu("google/reformer-enwik8")
    
