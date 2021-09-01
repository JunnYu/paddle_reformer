import torch
from transformers.models.reformer.modeling_reformer import (
    ReformerModel as PTReformerModel,
)
import os
import paddle
from reformer import ReformerModel as PDReformerModel
import shutil

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
    a = torch.from_numpy(a.cpu().numpy()).float()
    b = torch.from_numpy(b.cpu().numpy()).float()
    r1 = str((a - b).abs().mean())
    r2 = str((a - b).abs().max())
    return r1,r2

def test_gpu(model_path):
    paddle.set_device("gpu")
    shutil.rmtree("jilu",ignore_errors=True)
    os.makedirs("jilu",exist_ok=True)
    print(f"compare weights {model_path} on gpu")

    pd_inputs, pt_inputs = create_inputs(device="cuda")

    pd_model = PDReformerModel.from_pretrained(
        model_path
    )
    pd_model.eval()

    pd_outputs = pd_model(
        pd_inputs["input_ids"],
        attention_mask=pd_inputs["attention_mask"],
        output_hidden_states=True,
    )

    pt_model = PTReformerModel.from_pretrained(model_path)
    pt_model.eval()
    pt_model.cuda()


    pt_outputs = pt_model(
        pt_inputs["input_ids"],
        attention_mask=pt_inputs["attention_mask"],
        output_hidden_states=True,
    )
    print("compare hidden_states")
    for a, b in zip(pd_outputs.hidden_states, pt_outputs.hidden_states):
        r1,r2 = compare(a, b)
        print("mean dif: ", r1)
        print("max dif: ", r2)
        print("*"*50)
    print(len(os.listdir("jilu"))//2)
    with open(f"{model_path}.txt","w") as f:
        for i in range(0,  len(os.listdir("jilu"))//2):  # ~25
            qaz = torch.load(f"jilu/{i}.pt")
            wsx = paddle.load(f"jilu/{i}.pd")
            mean_value,max_value = compare(qaz,wsx)
            f.write(f"==========================================\n")
            f.write(f"{i}\n")
            f.write(f"mean dif: {mean_value} \n")
            f.write(f"max dif: {max_value}")
            f.write("\n")

            

if __name__ == "__main__":
    test_gpu("weights/12_layer")
