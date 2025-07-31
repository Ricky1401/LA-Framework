export NCCL_DEBUG=""

sudo apt-get update -y
sudo apt-get install python3-pip -y
sudo apt install python3.12-venv -y
python3 -m venv local_venv
source loacl_venv/bin/activate
sudo apt-get install git -y

pip3 install git+https://github.com/t1101675/transformers@minillm
pip3 install torch
pip3 install deepspeed
pip3 install numerize
pip3 install rouge-score
pip3 install torchtyping
pip3 install rich
pip3 install accelerate
pip3 install datasets
pip3 install peft

