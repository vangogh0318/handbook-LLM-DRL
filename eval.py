import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

model_path = "model/Qwen2.5-0.5B-Instruct"
model_path = "model/Qwen2.5-0.5B-RL-num_gen4_accreward/checkpoint-117"

print(f"model_path: {model_path}")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
system = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
# print(system)

while True:
    prompt = input("input:")
    text = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512, temperature=0.7)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("output:", response)
