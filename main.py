from distiller import Distiller
from quantizer import Quantizer
from argutils import generate_args
from evalutils import *


class Model:
    def __init__(self, name, checkpoint_path):
        self.name = name
        self.checkpoint_path = checkpoint_path

if __name__ == "__main__":
    args = generate_args()
    method = args.method # "quant" or "dist", depending on your use case

    if method == "quant":
        # Example for OPT
        """
        args.model_type = "opt"  # Example model type
        args.checkpoint_path = "./checkpoints/facebook-350m"  # Example checkpoint path
        args.dataset = "c4"  # Example dataset
        args.wbits = 4  # Example bits for quantization
        args.save = "./results/quantized_model/facebook-350m"  # Example save path
        """
        # Example for GPT-2
        args.model_type = "gpt2"
        args.checkpoint_path = "./checkpoints/gpt2-base"
        args.dataset = "c4"
        args.wbits = 4
        args.save = "./results/quantized_model/gpt2-base"
        

        quantizer = Quantizer(args)
        quantizer.quantize()

    elif method == "dist":
        # Example checkpoint paths (update as needed)
        teacher = Model("facebook-350m", "./checkpoints/facebook-350m")
        #teacher = Model("facebook-125m", "./checkpoints/facebook-125m")
        #student = Model("gpt2-base", "./checkpoints/gpt2-base")
        student = Model("facebook-125m", "./checkpoints/facebook-125m")

        distiller = Distiller(teacher, student)
        distiller.enable_sft_teacher()
        distiller.distill()

    elif method == "eval":
        # Example
        models = [("gpt2","./checkpoints/gpt2-base"),
                  ("gpt2","./results/quantized_model/gpt2-base"),
                  ("opt","./checkpoints/facebook-350m"),
                  ("opt","./results/quantized_model/facebook-350m"),
                  ("opt","./checkpoints/facebook-125m"),
                  ("opt","./results/distilled-facebook-125m/bs16-lr5e-06-G1-N1-NN1-lm1-len512/pe4_rs0.5_nr256_ln_sr_tm0.2/5000"),
                  ("opt","./results/quantized_model/facebook-125m")]
        
        for model_type, checkpoint_path in models:
            args.model_type = model_type
            args.checkpoint_path = checkpoint_path
            args.seqlen = 2048  # Example sequence length
            args.nsamples = 1000  # Example number of samples
            args.seed = 42  # Example seed for reproducibility
            eval_model(args)