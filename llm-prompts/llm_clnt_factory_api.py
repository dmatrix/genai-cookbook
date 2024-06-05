from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

import ollama

class ClientFactory:
    def __init__(self):
        self.clients = {}

    def register_client(self, client_name, client_class):
        self.clients[client_name] = client_class

    def create_client(self, client_name, **kwargs):
        client_class = self.clients.get(client_name)
        if client_class:
            if client_name == 'google':
                return client_class(kwargs['model_name'])
              
            else:
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
    elif isinstance(clnt, genai.GenerativeModel):
        chat_response = clnt.generate_content(user_content)
        return chat_response.text
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
    print("--------------------------")

    # Test Google Generative AI client
    client_factory.register_client('google', genai.GenerativeModel)
    client_type = 'google'
    client_kwargs = {"model_name": "gemini-1.5-flash",
                     "generation_config": {"temperature": 0.8,
                                          "api_key": "sk-1234567890abcdef1234567890abcdef",},
                     "system_instruction": "Please generate a polite to the user prompt. Do not make any answer",
    }

    client = client_factory.create_client(client_type, **client_kwargs)
    print(client)
    print("--------------------------")
    

