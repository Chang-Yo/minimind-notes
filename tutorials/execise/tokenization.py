from transformers import AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained("./model")

text="Hello, World!"
tokens=tokenizer.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Decoded: {tokenizer.decode(tokens)}")

message=[
    {"role":"user","content":"Hello"},
    {"role":"assistant","content":"Hello!"}
]
formatted=tokenizer.apply_chat_template(message,tokenize=False)
print(f"\nFormatted chat:\n{formatted}")

# Check vocabulary size
print(f"\nVocab size: {len(tokenizer)}")
print(f"Special tokens: {tokenizer.special_tokens_map}")