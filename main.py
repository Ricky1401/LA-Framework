from distiller import Distiller
from quantizer import Quantizer
from argutils import generate_args
from evalutils import *

from lib_gptq.gptq import * 
from lib_gptq.modelutils import *
from lib_gptq.quant import *



class Model:
    def __init__(self, name, checkpoint_path):
        self.name = name
        self.checkpoint_path = checkpoint_path

if __name__ == "__main__":
    args = generate_args()
    method = args.method # "quant" or "dist", depending on your use case

    if method == "quant":
        # Example for OPT
    
        args.model_type = "opt"  # Example model type
        args.checkpoint_path = "./checkpoints/facebook-350m"  # Example checkpoint path
        args.dataset = "c4"  # Example dataset
        args.wbits = 4  # Example bits for quantization
        args.save = "./results/quantized_model/facebook-350m"  # Example save path
        """
        # Example for GPT-2
        args.model_type = "gpt2"
        args.checkpoint_path = "./checkpoints/gpt2-base"
        args.dataset = "wikitext2"
        args.wbits = 4
        args.save = "./results/quantized_model/gpt2-base"
        """

        quantizer = Quantizer(args)
        quantizer.quantize()

    elif method == "dist":
        # Example checkpoint paths (update as needed)
        teacher = Model("gpt2-base", "./checkpoints/gpt2-base")
        #teacher = Model("facebook-125m", "./checkpoints/facebook-125m")
        #student = Model("gpt2-base", "./checkpoints/gpt2-base")
        student = Model("facebook-125m", "./checkpoints/facebook-125m")

        distiller = Distiller(teacher, student)
        distiller.enable_sft_teacher()
        distiller.distill()

    elif method == "eval":
        # Example for GPT-2 evaluation
        args.model_type = "gpt2"
        args.checkpoint_path = "./checkpoints/gpt2-base"
        args.dataset = "c4"
        args.seqlen = 1024  # Example sequence length
        args.nsamples = 1000  # Example number of samples
        args.seed = 42  # Example seed for reproducibility

        eval_model(args)