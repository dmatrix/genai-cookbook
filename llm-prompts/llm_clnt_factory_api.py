from openai import OpenAI
from anthropic import Anthropic
import ollama

class ClientFactory:
    def __init__(self):
        self.clients = {}

    def register_client(self, client_name, client_class):
        self.clients[client_name] = client_class

    def create_client(self, client_name, **kwargs):
        client_class = self.clients.get(client_name)
        if client_class:
            return client_class(**kwargs)
        else:
            raise ValueError(f"Client '{client_name}' not registered.")

def _get_chat_response(clnt: object, model: str, system_content: str, user_content:str) -> str:
    if isinstance(clnt, Anthropic):
        chat_response = clnt.messages.create(
                model=model,
                system = system_content,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=2500,
                temperature = 0.8)
        response = chat_response.content[0].text
        return response
    elif isinstance(clnt, OpenAI):
        chat_response = clnt.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}],
                temperature = 0.8)
        response = chat_response.choices[0].message.content
        return response
    elif isinstance(clnt, ollama.Client):
        chat_response = clnt.chat(model=model,
                messages=[{"role": "user", 
                           "content": user_content}])
        response = chat_response['message']['content']
        return response
    else:
        raise ValueError(f"Client '{clnt}' not registered or not supported.")
    
def get_commpletion(clnt: object, model: str, system_content: str, user_content:str) -> str:
    return _get_chat_response(clnt, model, system_content, user_content)

# Test ClientFactory
if __name__ == "__main__":

    # Test OpenAI client
    client_factory = ClientFactory()
    client_factory.register_client('openai', OpenAI)
    client_type = 'openai'
    client_kwargs = {"api_key": 
                        "sk-1234567890abcdef1234567890abcdef",
                    "base_url": 
                        "https://api.openai.com",}

    client = client_factory.create_client(client_type, **client_kwargs)
    print(client)
    print("--------------------------")

    # Test Anthropic client
    client_factory.register_client('anthropic', Anthropic)
    client_type = 'anthropic'
    client_kwargs = {"api_key": 
                        "sk-1234567890abcdef1234567890abcdef",}
                    
    client = client_factory.create_client(client_type, **client_kwargs)
    print(client)
    print("--------------------------")

    # Test Ollama client
    client_factory.register_client('ollama', ollama.Client)
    client_type = 'ollama'
    client_kwargs = {}
    client = client_factory.create_client(client_type, **client_kwargs) 
    print(client)
    

