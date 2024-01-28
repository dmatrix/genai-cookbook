# OpenAI and Anyscale Endpoints function calling.

<img src="./images/gpt_function_calling.png">

Function calling enhances the capabilities of Large Language Models (LLMs) in two main ways:

1. **Structured Responses**: LLMs can generate structured responses, like JSON objects, that can be used as arguments in subsequent functions within LLM applications. This is a more secured way
to ensure that your application is in control of executing a set
of functions for which the LLM has generated function arguments.

2. **Function Invocation**: Optionally, LLMs can directly invoke functions using these structured JSON objects as arguments. This allows them to utilize external tools such as Python interpreters, search websites, or access external databases, offering vast potention of possibilities.

However, direct function invocation by LLMs should be used cautiously,
as it can inadvertly call a funciton with generated arguments that 
you may not wish to invoke automatically (with undesirable consequences). For examplem, an LLM-generated e-mail response or deletion of a record response from an enterprise data source.

Use this extended and automatic LLM invocation of your desired 
functions judiciously. Nonetheless, both the aforementioned features
accord LLMs with extended functionality to build AI agents for automatic tasks.

To that extent, notebooks and examples in this section demonstrate:

* Creating or generating structured JSON for a downstream Python function within an LLM application
* Having the LLM invoke a Python function with JSON arguments and provide the results
* Generating SQL queries by asking LLM using natural language. These
queries are then executed, and its results fed back into LLM to generate
the final response to the user
* Using parallel function calling functionality using compatile OpenAI APIs

Explore the vast possibilities of function calling with OpenAI and Anyscale Endpoints