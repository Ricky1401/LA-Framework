from datautils import *


class Quantizer:
    def __init__(self, args, device="cuda"):
        self.model_type = args.model_type.lower()
        self.checkpoint_path = args.checkpoint_path
        self.device = device
        self.model = self._load_model()
        dataloader, testloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.checkpoint_path, seqlen=self.model.seqlen
        )
        self.dataloader = dataloader

    def _load_model(self):
        if self.model_type == "opt":
            from lib_gptq.opt import get_opt
            return get_opt(self.checkpoint_path).to(self.device)
        elif self.model_type == "llama":
            from lib_gptq.llama import get_llama
            return get_llama(self.checkpoint_path).to(self.device)
        elif self.model_type == "bloom":
            from lib_gptq.bloom import get_bloom
            return get_bloom(self.checkpoint_path).to(self.device)
        elif self.model_type == "gpt2":
            from transformers import GPT2LMHeadModel
            model = GPT2LMHeadModel.from_pretrained(self.checkpoint_path, torch_dtype='auto')
            model.seqlen = model.config.n_positions
            return model.to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        

    def quantize(self, dataloader, save_path):

        if self.model_type == "opt":
            from lib_gptq.opt import opt_sequential
            quantizers = opt_sequential(self.model, dataloader, self.device)
        elif self.model_type == "llama":
            from lib_gptq.llama import llama_sequential
            quantizers = llama_sequential(self.model, dataloader, self.device)
        elif self.model_type == "bloom":
            from lib_gptq.bloom import bloom_sequential
            quantizers = bloom_sequential(self.model, dataloader, self.device)
        elif self.model_type == "gpt2":
            # You need to implement gpt2_sequential if not present
            from lib_gptq.gpt2 import gpt2_sequential
            quantizers = gpt2_sequential(self.model, dataloader, self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Save the quantized model
        self.model.save_pretrained(save_path)
        print(f"Quantized model saved to {save_path}")