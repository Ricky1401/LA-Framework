import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

DEV = torch.device('cuda:0')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def find_layers_gpt(module, layers=[Conv1D, nn.Linear], name=''):
    res = {}

    # Match if module is an instance of any specified type
    if isinstance(module, tuple(layers)):
        res[name] = module

    # Recursively check children
    for name1, child in module.named_children():
        print(f"Checking child: {name1} of {name}")
        full_name = f"{name}.{name1}" if name else name1
        res.update(find_layers_gpt(child, layers=layers, name=full_name))

    return res

