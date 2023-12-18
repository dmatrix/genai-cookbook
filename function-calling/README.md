# OpenAI function calling.

Function calling enhances the capabilities of Large Language Models (LLMs) in two main ways:

1. **Structured Responses**: LLMs can generate structured responses, like JSON objects, that can be used as arguments in subsequent functions within LLM applications.

2. **Function Invocation**: Optionally, LLMs can directly invoke functions using these structured JSON objects as arguments. This allows them to utilize external tools such as Python interpreters, search websites, or access external databases, offering vast potention of possibilities.

However, direct function invocation by LLMs should be used cautiously.

In the notebook in this section, we demonstrate both:

* Creating or generating structured JSON for a downstream Python function within an LLM application.
* Having the LLM invoke a Python function with JSON arguments and provide the results.

<img src="./images/gpt_function_calling.png">

Explore the vast possibilities of function calling with OpenAI.