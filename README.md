# LA-Framework
This framework implement two libraries for compressing LLMs. It uses GPTQ for quantization and MiniLLM for distillation.


## Distillation - MiniLLM

To perform this:
- setup the environment
- download the checkpoint for the LLM teacher and student model
- download the "training/evaluation instruction-response data"
- process the data with the teacher model
- (optional) conduct SFT on the teacher model
- train the student model on (SFT) teacher model


```
bash setup_env.sh
huggingface-cli download gpt2 --repo-type model --local-dir ./checkpoints/gpt2-base
huggingface-cli download MiniLLM/dolly --repo-type dataset --local-dir ./data/dolly/
bash scripts/gpt2/tools/process_data_dolly.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/sft/sft_base.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/minillm/train_base_xl_no_pt.sh /PATH_TO/LMOps/minillm
```


## Quantization - GPTQ

To perform this:
- install also "transformers"
- load the model
- quantize




## References

- [GPTQ by IST-DASLab](https://github.com/IST-DASLab/gptq)  
  Efficient quantization techniques for LLMs, developed by IST-DASLab.

- [MiniLLM in LMOps by Microsoft](https://github.com/microsoft/LMOps/tree/main/minillm)  
  Lightweight LLM inference framework as part of Microsoft’s LMOps project.
