from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt = "One day, the developers become"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(input_ids, 
                        max_length=200, #long answer for story telling
                        num_return_sequences=1, 
                        temperature=0.2, #more controled and conservative answer
                        top_k=50,
                        top_p=0.95,
                        repetition_penalty=1.2,#avoid repitiion
                        do_sample=True)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text: ", generated_text)
