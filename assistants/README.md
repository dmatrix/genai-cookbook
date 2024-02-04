
# OpenAI Assistants APIs

The Assistants' API lets you create AI assistants in your applications. These assistants follow instructions and use models, tools, and knowledge to answer user questions. Currently, it supports tools like Code Interpreter, Retrieval (for files uploaded), and Function Calling. The OpenAI aims to add more tools and enable you to add your own tools to our platform in the future. This extends your LLM applications to interact, via tools, to external sources.

The architeture and data flow diagram below depicts the interaction among all
components that comprise OpenAI Assistant APIs. Central to understand is the 
Threads and Runtime that executes anyschronously, adding and reading messages
to the Threads.

<img src="./images/assistant_arch.png">

image inspired by[ source](https://www.youtube.com/watch?v=yzNG3NnF0YE)

In the notebooks in this secion, we will explore how you can use Assistants APIs for:
 * Retrievals (mini RAGs)
 * Functions calling
 * Parallel function calling with external tools
 * Code Interpreters for generating Python code
 * Custom code calling by assistants
 * Using extern webservices such as Google Search, etc

 | Notebook Description| Open with Colab |
|--------------------|-----------------|
| How to use OpenAI assistant tool retreiver - part_1| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmatrix/genai-cookbook/blob/main/assistants/1_how_to_use_assistant_tool_retriever_part_1.ipynb) |
| How to use OpenAI assistant tool retriever - part_2| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmatrix/genai-cookbook/blob/main/assistants/2_how_to_use_assistant_tool_retriever_part_2.ipynb) |
| How to use OpenAI assistant tool retreiver part_3| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmatrix/genai-cookbook/blob/main/assistants/3_how_to_use_assistant_tool_retriever_part_3.ipynb)|
| How to use OpenAI assistant tool calling web services | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmatrix/genai-cookbook/blob/main/assistants/4_how_to_use_parallel_function_calling.ipynb) |
| How to use OpenAI Assistant code interpreter| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmatrix/genai-cookbook/blob/main/assistants/5_how_to_use_code_interpreter.ipynb)|
| How to use OpenAI Assistant for custom code function calling| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmatrix/genai-cookbook/blob/main/assistants/6_how_to_use_function_calling_tool.ipynb) |
| How to use Open AI Assistant for custom Google searches|[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmatrix/genai-cookbook/blob/main/assistants/7_how_to_use_function_calling_tool_google_search.ipynb)| 