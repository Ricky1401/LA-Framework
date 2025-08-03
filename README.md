# LA-Framework

This framework provides an integrated interface for compressing Large Language Models (LLMs) using two state-of-the-art libraries:

- [GPTQ](https://github.com/IST-DASLab/gptq) for efficient quantization of LLMs.
- [MiniLLM](https://github.com/microsoft/LMOps/tree/main/minillm) for distillation and lightweight LLM inference.

## Features

- **Quantization**: Easily quantize OPT, LLaMA, BLOOM, and GPT-2 models using GPTQ, reducing memory and compute requirements for inference.
- **Distillation**: Distill large teacher models into smaller student models using MiniLLM, enabling efficient training and deployment.
- **Unified Interface**: Simple Python classes and scripts to perform both quantization and distillation with minimal code changes.

## Distillation - MiniLLM

This framework leverages MiniLLM for distillation:

1. Set up the environment and download teacher/student checkpoints.
2. Download the instruction-response data.
3. Optionally, fine-tune the teacher model.
4. Train the student model using the processed data.

Example workflow:

```bash
bash setup_env.sh
source local_venv/bin/activate
huggingface-cli download gpt2 --repo-type model --local-dir ./checkpoints/gpt2-base
huggingface-cli download facebook/opt-125m --repo-type model --local-dir ./checkpoints/facebook-125m
huggingface-cli download MiniLLM/dolly --repo-type dataset ./data/dolly/
python main.py dist
```

**Python usage example:**

You can also use the `Distiller` class directly in your main script to automate the distillation process:

```python
from distiller import Distiller

class Model:
    def __init__(self, name, checkpoint_path):
        self.name = name
        self.checkpoint_path = checkpoint_path

teacher = Model("gpt2-base", "./checkpoints/gpt2-base")
student = Model("facebook-125m", "./checkpoints/facebook-125m")
distiller = Distiller(teacher, student)
distiller.enable_sft_teacher()
distiller.distill()
```

## Quantization - GPTQ

This framework wraps GPTQ to support quantization of various LLM architectures. Example usage:

1. Install dependencies (including `transformers` and `gptq`).
2. Prepare your model checkpoint and calibration dataset.
3. Run quantization using the provided interface:

```python
from quantizer import Quantizer

args = ...  # Set up arguments (see main.py for example)
quantizer = Quantizer(args)
quantizer.quantize()
```

- Supports OPT, LLaMA, BLOOM, and GPT-2.
- Quantized models are saved and ready for efficient inference.

## References

- [GPTQ by IST-DASLab](https://github.com/IST-DASLab/gptq)  
  Efficient quantization techniques for LLMs.
- [MiniLLM in LMOps by Microsoft](https://github.com/microsoft/LMOps/tree/main/minillm)  
  Lightweight LLM distillation and inference framework.