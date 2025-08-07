import os
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM

class Distiller:
    def __init__(self, teacher_model, student_model):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.sft_teacher = False

    def check_models(self):
        """
        Check if the models exist in the checkpoints directory.
        Returns:
            bool: True if both models exist, False otherwise.
        """
        teacher_exists = os.path.exists(self.teacher_model.checkpoint_path)
        student_exists = os.path.exists(self.student_model.checkpoint_path)
        return teacher_exists and student_exists
    
    def enable_sft_teacher(self):
        """
        Enable the SFT (Supervised Fine-Tuning) mode for the teacher model.
        This is used to indicate that the teacher model should be trained in a supervised manner.
        """
        self.sft_teacher = True
        print("SFT mode enabled for teacher model.")

    def process_data_dolly(self):
        """
        Process the data for the Dolly dataset.
        """
        teacher_name = self.teacher_model.name
        print(f"Processing data for {teacher_name}")
        os.system(f"bash minillm/scripts/generic/tools/process_data_dolly.sh minillm {teacher_name}")
    
    def perform_sft_teacher(self):
        """
        Perform Supervised Fine-Tuning on the teacher model.
        """
        if not self.sft_teacher:
            return
        print(f"Supervised Fine-Tuning for {self.teacher_model.name} at {self.teacher_model.checkpoint_path}")
        #shutil.copytree(self.teacher_model.checkpoint_path, f"results/{self.teacher_model.name}/train/sft/e1-bs2-lr0.0005-G1-N1-NN1/5717", dirs_exist_ok=True)
        #os.system(f"bash minillm/scripts/generic/sft/sft_custom.sh minillm {self.teacher_model.name} {self.teacher_model.checkpoint_path}")
        self.teacher_model.checkpoint_path = f"results/{self.teacher_model.name}/train/sft/e1-bs2-lr0.0005-G1-N1-NN1/5717"

    def fix_vocabulary(self):
        """
        Loads tokenizers from two checkpoints, finds the smallest vocabulary size,
        resizes both models' embeddings, and saves the updated models and tokenizers
        back to their checkpoint directories.
        """
        # Load tokenizers
        ckpt_path1 = self.teacher_model.checkpoint_path
        ckpt_path2 = self.student_model.checkpoint_path
        save_path1 = str(ckpt_path1).replace("checkpoints", "results/resized_model")
        save_path2 = str(ckpt_path2).replace("checkpoints", "results/resized_model")

        tokenizer1 = AutoTokenizer.from_pretrained(ckpt_path1)
        tokenizer2 = AutoTokenizer.from_pretrained(ckpt_path2)

        # Find the smallest vocab size
        min_vocab_size = min(tokenizer1.vocab_size, tokenizer2.vocab_size)

        # Optionally, use the tokenizer with the smallest vocab for both
        if tokenizer1.vocab_size <= tokenizer2.vocab_size:
            shared_tokenizer = tokenizer1
        else:
            shared_tokenizer = tokenizer2

        # Load models
        model1 = AutoModelForCausalLM.from_pretrained(ckpt_path1)
        model2 = AutoModelForCausalLM.from_pretrained(ckpt_path2)

        # Resize embeddings
        model1.resize_token_embeddings(min_vocab_size)
        model2.resize_token_embeddings(min_vocab_size)

        # Save updated models and shared tokenizer
        model1.save_pretrained(save_path1)
        model2.save_pretrained(save_path2)
        shared_tokenizer.save_pretrained(save_path1)
        shared_tokenizer.save_pretrained(save_path2)

        self.teacher_model.checkpoint_path = save_path1
        self.student_model.checkpoint_path = save_path2

        print(f"Both models and tokenizers resized to vocab size {min_vocab_size} and saved.")


    def distill(self):
        """
        Perform distillation from teacher to student model.
        Returns:
            student_path: path to the stored results.
        """
        if not self.check_models():
            raise FileNotFoundError("One or both models do not exist in the checkpoints directory.")
        print(f"Distilling from {self.teacher_model.name} to {self.student_model.name}")
        self.process_data_dolly()
        self.perform_sft_teacher()
        self.fix_vocabulary()
        os.system(f"bash minillm/scripts/generic/minillm/train_custom.sh minillm {self.student_model.name} {self.student_model.checkpoint_path} {self.teacher_model.name} {self.teacher_model.checkpoint_path}")


        return "Bye"