import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def load_model_and_tokenizer(model_path):
    print(f"正在加载模型: {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, fast_tokenizer=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16, 
        device_map="auto",         
        trust_remote_code=True
    )
    model.eval()
    print("模型加载完成！")
    return model, tokenizer

def generate_reply(model, tokenizer, prompt, max_new_tokens=2048, temperature=0.2, top_p=0.95):
    formatted_prompt = f"[INST]{prompt}[/INST]\n[TAG]\ncorrect[/TAG]"
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(model.device)

    gen_cfg = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=1
    )

    print("正在生成回复...\n")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=gen_cfg,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = text.replace(formatted_prompt, "")
    return reply

if __name__ == "__main__":
    # 替换为你实际的模型路径
    model_path = ""
    
    model, tokenizer = load_model_and_tokenizer(model_path)
    user_prompt = "Generate a smart contract code snippet for an auction."
    response = generate_reply(model, tokenizer, user_prompt)
    print("=== 模型输出 ===")
    print(response)