import ollama

import sys 
sys.path.insert(0, "llm-prompts")
from  llm_clnt_factory_api import ClientFactory, get_commpletion

# Chat with the default model
contents = [
            "Why did the chicken cross the road?",
            "Write a haiku about the ocean",
            "What is the meaning of life?",
            ]

ROLE = 'user'
clnt = ollama.Client()

# create an Anthropic client instance using our
# client factory method 
client_factory = ClientFactory()
client_type = "ollam"
client_factory.register_client(client_type, ollama.Client)
client_kwargs = {}
clnt = client_factory.create_client(client_type, **client_kwargs)

for content in contents:
    # get the response
    BOLD_BEGIN = "\033[1m"
    BOLD_END   =   "\033[0m"
    response = get_commpletion(clnt, 
                               "mistral", "user", content)
    print(f"\n{BOLD_BEGIN}Prompt:{BOLD_END} {content}")
    print(f"\n{BOLD_BEGIN}Answer:{BOLD_END} {response}")
    print(f"User: {content}")
    print(f"Ollama: {response}")
    print("-------------------")
