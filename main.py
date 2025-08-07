from distiller import Distiller
from quantizer import Quantizer
from argutils import generate_args

class Model:
    def __init__(self, name, checkpoint_path):
        self.name = name
        self.checkpoint_path = checkpoint_path

if __name__ == "__main__":
    args = generate_args()
    method = args.method # "quant" or "dist", depending on your use case

    if method == "quant":
        args = generate_args()
        args.model_type = "opt"  # Example model type
        args.checkpoint_path = "./checkpoints/facebook-125m"  # Example checkpoint path
        args.dataset = "c4"  # Example dataset
        args.wbits = 4  # Example bits for quantization
        args.save = "./results/quantized_model/facebook-125m"  # Example save path

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