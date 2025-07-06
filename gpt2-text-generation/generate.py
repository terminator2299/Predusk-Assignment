from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_text(prompt, temperature, k=50, max_length=50):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Set pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    attention_mask = inputs['attention_mask']

    output = model.generate(
        inputs['input_ids'],
        attention_mask=attention_mask,
        max_length=len(inputs['input_ids'][0]) + max_length,
        do_sample=True,
        top_k=k,
        temperature=temperature,
        pad_token_id=tokenizer.pad_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

prompt = "Once upon a time"
output_07 = generate_text(prompt, temperature=0.7)
output_10 = generate_text(prompt, temperature=1.0)

with open("generated_outputs.txt", "w") as f:
    f.write("=== Output with temperature=0.7 ===\n")
    f.write(output_07 + "\n\n")
    f.write("=== Output with temperature=1.0 ===\n")
    f.write(output_10 + "\n")

print("✅ Done! Check 'generated_outputs.txt'.")
