import torch
from transformers.models.reformer.modeling_reformer import (
    ReformerModelWithLMHead as PTReformerModelWithLMHead,
)

import paddle
from reformer import ReformerModelWithLMHead as PDReformerModelWithLMHead


def create_buckets():
    shape = (2, 64, 1, 96)
    tensor = torch.randn(shape)
    torch.save(tensor, "buckets/bucks.pt")
    paddle.save(paddle.to_tensor(tensor.numpy()), "buckets/bucks.pd")


def compare(a, b):
    if isinstance(a, int):
        print(a == b)
        return
    if isinstance(a, float):
        print(a == b)
        return
    a = torch.from_numpy(a.detach().numpy()).float()
    b = torch.from_numpy(b.detach().numpy()).float()
    print("mean difference:", (a - b).abs().mean())
    print("max difference:", (a - b).abs().max())


def create_inputs(device="cpu"):
    pdx = {
        "input_ids": paddle.randint(0, 32, shape=[1, 64 * 128]),
        "attention_mask": paddle.randint(0, 2, shape=[1, 64 * 128]),
    }
    ptx = {k: torch.from_numpy(v.numpy()).to(device) for k, v in pdx.items()}

    return pdx, ptx


def test_cpu():
    paddle.set_device("cpu")
    # 固定buckets
    create_buckets()
    pd_model = PDReformerModelWithLMHead.from_pretrained(
        "reformer-crime-and-punishment-64-128"
    )
    pd_model.train()
    pt_model = PTReformerModelWithLMHead.from_pretrained(
        "reformer-crime-and-punishment-64-128"
    )
    pt_model.train()
    pd_inputs, pt_inputs = create_inputs(device="cpu")
    pd_outputs = pd_model(
        pd_inputs["input_ids"],
        attention_mask=pd_inputs["attention_mask"],
        labels=pd_inputs["input_ids"],
    )
    pt_outputs = pt_model(
        pt_inputs["input_ids"],
        attention_mask=pt_inputs["attention_mask"],
        labels=pt_inputs["input_ids"],
    )
    print("compare loss")
    compare(pd_outputs.loss, pt_outputs.loss)
    # 反向传播
    pd_outputs.loss.backward()
    pt_outputs.loss.backward()
    print("=" * 50)
    print("compare grad")
    compare(
        pd_model.get_input_embeddings().weight.grad,
        pt_model.get_input_embeddings().weight.grad,
    )


def test_gpu():
    # 固定buckets
    paddle.set_device("gpu")
    create_buckets()
    pd_model = PDReformerModelWithLMHead.from_pretrained(
        "reformer-crime-and-punishment-64-128"
    )
    pd_model.train()
    pt_model = PTReformerModelWithLMHead.from_pretrained(
        "reformer-crime-and-punishment-64-128"
    )
    pt_model.to("cuda:0")
    pt_model.train()
    pd_inputs, pt_inputs = create_inputs(device="cuda:0")
    pd_outputs = pd_model(
        pd_inputs["input_ids"],
        attention_mask=pd_inputs["attention_mask"],
        labels=pd_inputs["input_ids"],
    )
    pt_outputs = pt_model(
        pt_inputs["input_ids"],
        attention_mask=pt_inputs["attention_mask"],
        labels=pt_inputs["input_ids"],
    )
    print("compare loss")
    compare(pd_outputs.loss, pt_outputs.loss.cpu())
    # 反向传播
    pd_outputs.loss.backward()
    pt_outputs.loss.backward()
    print("=" * 50)
    print("compare grad")
    compare(
        pd_model.get_input_embeddings().weight.grad,
        pt_model.get_input_embeddings().weight.grad.cpu(),
    )


if __name__ == "__main__":
    print("test on cpu!")
    test_cpu()
    print("=" * 50)
    print("test on gpu!")
    test_gpu()
