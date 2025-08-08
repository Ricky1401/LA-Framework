# LA-Framework

This framework provides an integrated interface for compressing Large Language Models (LLMs) using two state-of-the-art libraries:

- [GPTQ](https://github.com/IST-DASLab/gptq) for efficient quantization of LLMs.
- [MiniLLM](https://github.com/microsoft/LMOps/tree/main/minillm) for distillation and lightweight LLM inference.

## Features

- **Quantization**: Easily quantize OPT, LLaMA, BLOOM, and GPT-2 models using GPTQ, reducing memory and compute requirements for inference.
- **Distillation**: Distill large teacher models into smaller student models using MiniLLM, enabling efficient training and deployment.
- **Unified Interface**: Simple Python classes and scripts to perform both quantization and distillation with minimal code changes.

> **Note:** Both quantization and distillation require a CUDA-capable GPU and the CUDA toolkit/libraries to be installed.

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
- Instantiates the teacher and student models based on the specified checkpoint paths.
- Creates a `Distiller` object with these models.
- Enables supervised fine-tuning for the teacher (if needed).
- Calls the `distill()` method to perform the full distillation workflow automatically.

This allows you to launch the entire distillation process with a single command.

## Quantization - GPTQ

This framework wraps GPTQ to support quantization of various LLM architectures. Example usage:

1. Install dependencies (you can use the previous enviroment by including also `transformers`).
2. Download your model checkpoint.
3. Run quantization using the provided interface:

```python
from quantizer import Quantizer

args = ...  # Set up arguments (see main.py for example)
quantizer = Quantizer(args)
quantizer.quantize()
```

- Supports OPT, LLaMA, BLOOM, and GPT-2.
- Quantized models are saved and ready for efficient inference.

## Experimental Evaluation

To test the performance of the quantization and distillation method, I used the **GPTQ** built-in evaluation function, which estimates the perplexity of the models on the `c4` and `wikitext2` datasets.

In addition, I employed the [LLMCBench](https://github.com/AboveParadise/LLMCBench/) suite to perform the **MMLU** accuracy benchmark. The results are shown below:

| Model                  | c4 Perplexity | wt2 Perplexity | MMLU    |
|------------------------|--------------:|---------------:|--------:|
| GPT-2 base             | 29.706        | 29.951         | 0.2299  |
| GPT-2 quant.            | 32.029        | 33.149         | 0.2302  |
| OPT-350m               | 20.929        | 22.005         | 0.2418  |
| OPT-350m quant.         | 22.846        | 24.022         | 0.2384  |
| OPT-125m               | 24.714        | 27.653         | 0.2290  |
| OPT-350m → 125m dist.  | 25.556        | 29.919         | 0.2296  |
| OPT-125m quant.         | 27.346        | 31.069         | 0.2417  |


## References

- [GPTQ by IST-DASLab](https://github.com/IST-DASLab/gptq)  
  Efficient quantization techniques for LLMs.
- [MiniLLM in LMOps by Microsoft](https://github.com/microsoft/LMOps/tree/main/minillm)  
  Lightweight LLM distillation and inference framework.
- [LLMCBench](https://github.com/AboveParadise/LLMCBench/)  
  Rigorously designed benchmark with an in-depth analysis for LLM compression algorithms.