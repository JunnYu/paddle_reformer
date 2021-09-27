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
    pdx = {
        "input_ids": paddle.randint(0, 32, shape=[4, 128]),
        "attention_mask": paddle.randint(0, 2, shape=[4, 128]),
    }
    ptx = {k: torch.from_numpy(v.numpy()).to(device) for k, v in pdx.items()}
    return pdx, ptx


def compare(a, b):
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    print("mean dif:", np.abs(a - b).mean())
    print("max dif:", np.abs(a - b).max())

def write_compare(a, b):
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    return str(np.abs(a - b).mean()),str(np.abs(a - b).max())

def test_gpu(model_path):
    try:
        shutil.rmtree("buckets")
        shutil.rmtree("jilu")
    except:
        pass
    os.makedirs("buckets")
    os.makedirs("jilu")
    paddle.set_device("gpu")
    paddle.set_default_dtype("float64")
    print(f"compare weights {model_path} on gpu")
    pd_model = PDReformerModelWithLMHead.from_pretrained(
        model_path
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
    with open("short_result.txt","w") as f:
        for i in range(0,  len(os.listdir("jilu"))//2):  # ~25
            qaz = torch.load(f"jilu/{i}.pt")
            wsx = paddle.load(f"jilu/{i}.pd")
            mean_value,max_value = write_compare(qaz,wsx)
            f.write(f"==========================================\n")
            f.write(f"{i}\n")
            f.write(f"mean dif: {mean_value} \n")
            f.write(f"max dif: {max_value}")
            f.write("\n")

if __name__ == "__main__":
    test_gpu("weights/12_layer")
