import torch
import torch.nn as nn

from lib_gptq.datautils import *

from lib_gptq.gpt2 import gpt2_eval, get_gpt2
from lib_gptq.opt import opt_eval, get_opt

def eval_model(args):
    if args.model_type == 'opt':
        model = get_opt(args.checkpoint_path)
        datasets = ['c4', 'wikitext2'] 
        for dataset in datasets: 
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.checkpoint_path, seqlen=args.seqlen
            )
            print(dataset)
            opt_eval(model, testloader, torch.device('cuda:0'))

    elif args.model_type == 'gpt2':
        model = get_gpt2(args.checkpoint_path)
        datasets = ['c4', 'wikitext2'] 
        for dataset in datasets: 
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.checkpoint_path, seqlen=args.seqlen
            )
            print(dataset)
            gpt2_eval(model, testloader, torch.device('cuda:0'))
    else:
        raise ValueError(f"Unsupported model type for evaluation: {model.model_type}")