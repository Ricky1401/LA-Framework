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

When you run `python main.py dist`, the main script:
- Instantiates the teacher and student models with their checkpoint paths.
- Creates a Distiller object with these models.
- Enables supervised fine-tuning for the teacher (if needed).
- Calls the distill() method to perform the full distillation workflow automatically.

This allows you to launch the entire distillation process with a single command.

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