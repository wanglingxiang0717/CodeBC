import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "base_model_path"
lora = "lora_path" 
out  = "output_path"

tok = AutoTokenizer.from_pretrained(base, use_fast=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base, 
    torch_dtype=torch.bfloat16, 
    device_map="cpu" 
)

print("Base model loaded.")

model = PeftModel.from_pretrained(base_model, lora)
print("LoRA weights loaded successfully.")

model = model.merge_and_unload()
print("Model merged.")

os.makedirs(out, exist_ok=True)
model.save_pretrained(out)
tok.save_pretrained(out)
print("合并完成，保存至：", out)