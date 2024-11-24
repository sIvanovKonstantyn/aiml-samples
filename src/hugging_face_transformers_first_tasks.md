```
>pip install transformers
```

----------------------------  

Play with pretrained Bert model: 
```
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, worlds!", return_tensors="pt")
outputs = model(**inputs)

print("result: ", outputs.last_hidden_state)
```

----------------------------  

Play with pretrained GPT model: 

```
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def chat_bot(input_text):
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

print(chat_bot("Hello, tell a joke"))
print(chat_bot("Hello, how are you?"))
```
