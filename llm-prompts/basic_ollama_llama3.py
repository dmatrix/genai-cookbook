import ollama

response = ollama.generate(model="llama3", prompt="Why is the sky blue?", stream=True)
# Stream response
for chunk in response:
    data = chunk["response"]
    print(data, end="")