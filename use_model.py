import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def main():

    checkpoints_dir = input("Enter the path to the checkpoints folder: ").strip()
    if not os.path.isdir(checkpoints_dir):
        print("Invalid checkpoints folder.")
        sys.exit(1)

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoints_dir)
    model = AutoModelForCausalLM.from_pretrained(checkpoints_dir)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    print("Type your questions. Type 'stop' to exit.")
    while True:
        question = input("You: ").strip()
        if question.lower() == "stop":
            break
        result = generator(question, max_new_tokens=128, do_sample=True, temperature=0.7)
        print("Model:", result[0]['generated_text'][len(question):].strip())

if __name__ == "__main__":
    main()