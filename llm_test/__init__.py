from transformers import AutoModelForCausalLM, AutoTokenizer

# from transformers import NVLM_D
# model = NVLM_D.from_pretrained(nvidia/NVLM-D-72B, trust_remote_code=True)

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chat_with_bot():
    while True:
        user_input = input("> ")
        inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        # generated_text[len(user_input):].strip()


if __name__ == '__main__':
    chat_with_bot()