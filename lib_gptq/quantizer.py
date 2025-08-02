from lib_gptq.datautils import *


class Quantizer:
    def __init__(self, args, device="cuda"):
        self.args = args
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
            from lib_gptq.gpt2 import get_gpt2
            return get_gpt2(self.checkpoint_path).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        

    def quantize(self):
        dataloader = self.dataloader
        save_path = self.args.save
        if self.model_type == "opt":
            from lib_gptq.opt import opt_sequential_ext
            quantizers = opt_sequential_ext(self.args, self.model, dataloader, self.device)
        elif self.model_type == "llama":
            from lib_gptq.llama import llama_sequential_ext
            quantizers = llama_sequential_ext(self.args, self.model, dataloader, self.device)
        elif self.model_type == "bloom":
            from lib_gptq.bloom import bloom_sequential_ext
            quantizers = bloom_sequential_ext(self.args, self.model, dataloader, self.device)
        elif self.model_type == "gpt2":
            from lib_gptq.gpt2 import gpt2_sequential_ext
            quantizers = gpt2_sequential_ext(self.args, self.model, dataloader, self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Save the quantized model
        self.model.save_pretrained(save_path)
        print(f"Quantized model saved to {save_path}")