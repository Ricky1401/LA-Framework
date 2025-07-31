import os

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

    def process_data_dolly(self, teacher_name):
        """
        Process the data for the Dolly dataset.
        Args:
            teacher_name (str): Name of the teacher model.
        Returns:
            str: Path to the processed data.
        """
        # Example placeholder logic
        os.system(f"bash minillm/scripts/generic/tools/process_data_dolly.sh . {teacher_name}")
        print(f"Processing data for {teacher_name}")
        return f"data/processed_data/dolly/*/{teacher_name}"

    def distill(self):
        """
        Perform distillation from teacher to student model.
        Returns:
            student_path: path to the stored results.
        """
        if not self.check_models():
            raise FileNotFoundError("One or both models do not exist in the checkpoints directory.")
        print(f"Distilling from {self.teacher_model.name} to {self.student_model.name}")
        self.process_data_dolly(self.teacher_model.name)


        return "Bye"