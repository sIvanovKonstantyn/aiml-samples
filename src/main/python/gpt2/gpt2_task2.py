from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt = "One day, the developers become"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

current_temperature = 0.5
for i in range(1,5):
    outputs = model.generate(input_ids, 
                        max_length=50, 
                        num_return_sequences=10, 
                        temperature=current_temperature, 
                        do_sample=True,
                       top_k=50,
                       top_p=0.9)

    
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output[i], skip_special_tokens=True)
        print(f"Templerature: {current_temperature}. Generated text {i + 1}: {generated_text}")

    current_temperature = current_temperature + 0.25
