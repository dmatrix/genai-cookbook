# GenAI Cookbook

<a href="https://github.com/dmatrix/genai-cookbook"><img src="https://img.shields.io/github/stars/dmatrix/genai-cookbook" alt="github-stars"></a>
<a href="https://twitter.com/2twitme"><img src="https://img.shields.io/twitter/follow/2twitme?label=Follow" alt="twitter"></a>

<img src=images/gen_ai_cookbook_img_1.png>

Welcome to a mixture of Gen AI cookbook **how-to** recipes for Gen AI applications. These simple guides span across get-started examples on LLM prompting strategies, DSPy, building simple chatbots, retrieval generation augmentation (RAGs), LLM-based applications on OpenAI, Anthropic, Gemini, or Anyscale Endpoints, etc. 

Aimed to develpers who are begining their journey on LLMs, you'll find examples code and/or guides for common tasks using the OpenAI API, Anthropic, Google Gemini, Pinecone, Anyscale Endpoints and [Ray framework](https://www.ray.io/). 

To try these examples, you'll need an [OpenAI](https://platform.openai.com/docs/introduction) account and an associated API key, [Anthropic](https://docs.anthropic.com/claude/docs/intro-to-claude), [Pinecone](https://www.pinecone.io/pricing/), or an [Anyscale Endpoint](https://www.anyscale.com/get-started) account and an associated key. Even better, install [OLlama](https://ollama.com/) on your laptop. 

All examples are in Python, yet the concepts can be adapted to any programming language, so long as you can make REST API calls in the target language or use target language's SDK.

The examples in this cook are inspired (some modified, expanded or copied) from these resources, including:

 * [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
 * [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
 * [Prompt Engineering Guide](https://www.promptingguide.ai/introduction) and [Prompt Engineering course](https://maven.com/dair-ai/prompt-engineering-llms?promoCode=MAVENMONDAY) by Elvis Saravia
 * [How I Won Singapore's GPT-4 Prompt Engineering Competition](https://towardsdatascience.com/how-i-won-singapores-gpt-4-prompt-engineering-competition-34c195a93d41) by Sheila Teo
 * [Ray Documentation](https://docs.ray.io/en/latest/) and [Anyscale blogs](https://www.anyscale.com/blog)
 * [Anyscale Endpoints Documentation](https://docs.endpoints.anyscale.com/)
 * [Anthropic Developer and User Guide](https://docs.anthropic.com/claude/docs/intro-to-claude)
 * [ChatGPT Prompt Engineering for Developers](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction)
 * [LangChain for LLM Application Development](https://learn.deeplearning.ai/langchain/lesson/1/introduction)
 * [Building Systems with the ChatGPT API](https://learn.deeplearning.ai/chatgpt-building-system/lesson/1/introduction)
 * [DSPy: Programming framework for Language Models](https://dspy-docs.vercel.app/docs/intro)
 * Various blog posts on medium.

 ## Directory Structure or Chapters for the Book

| Directory Names| Description | 
|---------------|-------------|
| genai_intro | General Introduction to GenAI, Foundation Models, LLMS |
| agents| What are agents, agent archicture, how to write agents|
| assistants| How to work with OpenAI Assistants as agents |
| chatbots| How to write simple chatbots with LLMs |
| dspy| Quick overview: what and why and how to use DSPy|
|embeddings-and-vectordbs| Introduction to vector embeddings and how they play a role in semantic searches for LLM Gen apps|
|evaluation| Evaluation LLM is not easy. How to use exprimentaiton with MLflow tracking and tracing to evaluate LLMs|
|fine-tuning | Why and when to fine-tune and how. |
|function-calling|How to use both OpenAI and OSS LLM funciton calling to extend LLM application functionality|
| genai_intro | General Introduction to GenAI, 
Foundation Models, LLMS |
| gpts| A walk through OpenAI's GPT models and how and when to use them|
| llm-prompts|An introduction to myriad LLM engineering techniques using clsoed and open source LLM models|
|rags|Retrieval Augmentation Retrieval is the TayLor Swift of LLM applications; everyone wants; everyone builds them. A techincal of types of RAGS, how to to implement them|

 ## Current Blogs on this cookbook chapters

 * [Best Prompt Techniques for Best LLM Responses](https://medium.com/the-modern-scientist/best-prompt-techniques-for-best-llm-responses-24d2ff4f6bca)
* [LLM Beyond its Core Capabilities as AI Assistants or Agents](https://medium.com/@2twitme/llm-beyond-its-core-capabilities-as-ai-assistants-or-agents-704ffb972934)
* [Crafting Intelligent User Experiences: A Deep Dive into OpenAI Assistants API](https://medium.com/@2twitme/crafting-intelligent-user-experiences-a-deep-dive-into-openai-assistants-api-00439ace108a)
* [An Intuitive 101 Guide to Vector Embeddings](https://medium.com/@2twitme/an-intuitive-101-guide-to-vector-embeddings-ffde295c3558)
* [An Exploratory Tour of Retrieval Augmented Generation (RAG) Paradigm](https://medium.com/@2twitme/an-exploratory-tour-of-retrieval-augmented-generation-rag-paradigm-3940c1947d27)
* [An Exploratory Tour of DSPy: A Framework for Programing Language Models, not Prompting](https://medium.com/@2twitme/an-exploratory-tour-of-dspy-a-framework-for-programing-language-models-not-prompting-711bc4a56376)

## Environment files
Since I use either OpenAI, Anthropic, or Anyscale Endpoints, I provide three environment template files: *env_anyscale_template*, *env_anthropic_template*, and *env_openai_template*. Add your keys and model name to either of the files and copy the file to **.env** to the top-level directory. To migrate any OpenAI code to Anyscale Endpoints, use this [simple guide](https://docs.endpoints.anyscale.com/guides/migrate-from-openai/). For the most part, the migration is seamless.

Also, you'll require some Python libraries. Use `pip install -r requirements.txt` to install them.

**Note**: Don't share your environment files with API keys publicly.

## Contributing
I welcome contributions. Let's make this a Generative AI developer community-driven resource. Your contributions can include additions or expansions of these how-to guides.

Check existing [issues](https://github.com/dmatrix/genai-cookbook/issues) and [pull requests](https://github.com/dmatrix/genai-cookbook/pulls) before contributing to avoid duplication. If you have suggestions for examples or guides, share them on the issues page.


Have GenAI fun! 🥳️

Jules
