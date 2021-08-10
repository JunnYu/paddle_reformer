import torch
from transformers.models.reformer.modeling_reformer import (
    ReformerModelWithLMHead as PTReformerModelWithLMHead,
)

import paddle
from reformer import ReformerModelWithLMHead as PDReformerModelWithLMHead

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
    a = torch.from_numpy(a.numpy())
    b = torch.from_numpy(b.numpy())
    print("mean dif:", (a - b).abs().mean())
    print("max dif:", (a - b).abs().max())


def test_cpu(model_path):
    print(f"compare weights {model_path} on cpu")
    paddle.set_device("cpu")
    pd_model = PDReformerModelWithLMHead.from_pretrained(
        model_path.replace("google", "paddle")
    )
    pd_model.eval()
    pt_model = PTReformerModelWithLMHead.from_pretrained(model_path)
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
    paddle.set_device("gpu")
    print(f"compare weights {model_path} on gpu")
    pd_model = PDReformerModelWithLMHead.from_pretrained(
        model_path.replace("google", "paddle")
    )
    pd_model.eval()
    pt_model = PTReformerModelWithLMHead.from_pretrained(model_path)
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
