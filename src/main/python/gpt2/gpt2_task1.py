from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt = "One day, the developers become"
input_ids = tokenizer.encode(prompt, return_tensors="pt")#pyTorch tensor format

output = model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=0.7, do_sample=True)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text: ", generated_text)
