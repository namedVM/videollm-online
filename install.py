from time import sleep

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/meta-llama/Meta-Llama-3-8B-Instructdatasets",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,  # 强制只从本地读取，不尝试连接 HF
)
model = PeftModel.from_pretrained(
    base_model,
    "chenjoya/videollm-online-8b-v1plus",
    ensure_weight_tying=True,
)
model = model.merge_and_unload()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model loaded successfully, waitting...")
sleep(1000000)
