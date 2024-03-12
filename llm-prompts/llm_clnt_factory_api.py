from openai import OpenAI
from anthropic import Anthropic

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

# Test ClientFactory
if __name__ == "__main__":
    client_factory = ClientFactory()
    client_factory.register_client('openai', OpenAI)
    client_factory.register_client('anthropic', Anthropic)
    client_type = 'openai'

    client_kwargs = {"api_key": 
                        "sk-1234567890abcdef1234567890abcdef",
                    "base_url": "https://api.openai.com",}

    client = client_factory.create_client(client_type, **client_kwargs)
    print(client)

    print("--------------------------")
    client_type = 'anthropic'
    client_kwargs = {"api_key": 
                        "sk-1234567890abcdef1234567890abcdef",}
                    
    client = client_factory.create_client(client_type, **client_kwargs)
    print(client)

client_factory = ClientFactory()
client_factory.register_client('openai', OpenAI)
client_factory.register_client('anthropic', Anthropic)
client_type = 'openai'

client_kwargs = {"api_key": 
                    "sk-1234567890abcdef1234567890abcdef",
                "base_url": "https://api.openai.com",}

client = client_factory.create_client(client_type, **client_kwargs)
print(client)

print("--------------------------")
client_type = 'anthropic'
client_kwargs = {"api_key": 
                    "sk-1234567890abcdef1234567890abcdef",}
                
client = client_factory.create_client(client_type, **client_kwargs)
print(client)


