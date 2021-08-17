import json

from reformer.modeling import ReformerModel as PDReformerModel
from transformers.models.reformer.modeling_reformer import (
    ReformerModel as PTReformerModel,
)
from transformers import ReformerConfig
import paddle
from collections import OrderedDict

paddle.set_device("cpu")


def load_json(file_path):
    with open(str(file_path), "r", encoding="utf8") as f:
        data = json.load(f)
    return data


def to_paddle(x):
    return paddle.to_tensor(x.numpy())


def generate_random_2_layer():
    pdjson = load_json("weights/random_2_layer/model_config.json")
    pd_model = PDReformerModel(**pdjson)
    c = ReformerConfig.from_json_file("weights/random_2_layer/config.json")
    pt_model = PTReformerModel(c)
    pd_model.eval()
    pt_model.eval()
    d = OrderedDict()
    for k, v in pt_model.state_dict().items():
        if "_mask_value_" in k:
            continue
        if v.ndim == 2 and "embedding" not in k:
            v = v.transpose(1, 0)
        d[k] = to_paddle(v)
    pd_model.set_dict(d)
    pd_model.save_pretrained("weights/random_2_layer")
    pt_model.save_pretrained("weights/random_2_layer")


def generate_random_12_layer():
    pdjson = load_json("weights/random_12_layer/model_config.json")
    pd_model = PDReformerModel(**pdjson)
    c = ReformerConfig.from_json_file("weights/random_12_layer/config.json")
    pt_model = PTReformerModel(c)
    pd_model.eval()
    pt_model.eval()
    d = OrderedDict()
    for k, v in pt_model.state_dict().items():
        if "_mask_value_" in k:
            continue
        if v.ndim == 2 and "embedding" not in k:
            v = v.transpose(1, 0)
        d[k] = to_paddle(v)
    pd_model.set_dict(d)

    pd_model.save_pretrained("weights/random_12_layer")
    pt_model.save_pretrained("weights/random_12_layer")


if __name__ == "__main__":
    generate_random_2_layer()
    generate_random_12_layer()