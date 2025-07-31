from distiller import Distiller

class Model:
    def __init__(self, name, checkpoint_path):
        self.name = name
        self.checkpoint_path = checkpoint_path

if __name__ == "__main__":
    # Example checkpoint paths (update as needed)
    teacher = Model("gpt2-base", "./checkpoints/gpt2-base")
    student = Model("facebook/125m", "./checkpoints/facebook-125m")

    distiller = Distiller(teacher, student)
    distiller.enable_sft_teacher()
    distiller.distill()