from distiller import Distiller
from lib_gptq.quantizer import Quantizer

class Model:
    def __init__(self, name, checkpoint_path):
        self.name = name
        self.checkpoint_path = checkpoint_path

def generate_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_type', type=str,
        choices=['opt', 'llama', 'bloom', 'gpt2'],
        help='Type of model to load; possible [opt,llama,bloom,gtp2].'
    )
    parser.add_argument(
        '--checkpoint_path', type=str,
        help='Checkpoint of the model to load.'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Whether to compute perplexity during benchmarking for verification.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--faster-kernel', action='store_true',
        help='Whether to use the new faster kernel for benchmarking.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    method = "quantization"  # or "distillation", depending on your use case

    if method == "quantization":
        args = generate_args()
        args.model_type = "opt"  # Example model type
        args.checkpoint_path = "./checkpoints/facebook-125m"  # Example checkpoint path
        args.dataset = "c4"  # Example dataset
        args.wbits = 4  # Example bits for quantization
        args.save = "./results/quantized_model/facebook-125m"  # Example save path

        quantizer = Quantizer(args)
        quantizer.quantize()

    elif method == "distillation":
        # Example checkpoint paths (update as needed)
        teacher = Model("gpt2-base", "./checkpoints/gpt2-base")
        #teacher = Model("facebook-125m", "./checkpoints/facebook-125m")
        #student = Model("gpt2-base", "./checkpoints/gpt2-base")
        student = Model("facebook-125m", "./checkpoints/facebook-125m")

        distiller = Distiller(teacher, student)
        distiller.enable_sft_teacher()
        distiller.distill()