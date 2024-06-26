# GenAI Cookbook

Generative Artificial Intelligence (GenAI) is introducing novelty, creativity, and productivity in various domains of human activities. It's novel because it responds to human-like natural language instructions; creative because it can generate new human-like content such as text, audio, images, and video, from existing trained data, unimaginalbe before; productive because it's faster than humans at completing tasks or activities.

<a href="https://github.com/dmatrix/genai-cookbook"><img src="https://img.shields.io/github/stars/dmatrix/genai-cookbook" alt="github-stars"></a>
<a href="https://twitter.com/2twitme"><img src="https://img.shields.io/twitter/follow/2twitme?label=Follow" alt="twitter"></a>

<img src=images/gen_ai_cookbook_img_1.png>

So welcome to a mixture of Gen AI cookbook **how-to** recipes for Gen AI applications. Mixture because GenAI is a larger umbrella conceptual term encompassing myriad natural language processing (NLP) applications built using language models. 

These simple guides for these applications span across get-started examples on using LLM prompting strategies; exloring DSPy framework as an alternative to prompt engineering as declartive and programmatic ways to program LLMs; building simple chatbots; implementing retrieval generation augmentation (RAGs), incorporating personal or orgainzational data;  fine-tuning LLMs for domain specific tasks; extending LLM functionality with agents and function; and employing language models from OpenAI, Anthropic, Gemini, Anyscale Endpoints, OLlama, and Databricks platform, etc. 

Aimed at beginer and intermediate developers who are embarking on their journey on GenAI, examples code and how-to guides exemplify how to use the OpenAI API, Anthropic, Open Source Models, Google Gemini, Pinecone, Anyscale Endpoints, Databricks, and [Ray framework](https://www.ray.io/). 

To try these examples, you'll need an [OpenAI](https://platform.openai.com/docs/introduction) account and an associated API key, [Anthropic](https://docs.anthropic.com/claude/docs/intro-to-claude), [Pinecone](https://www.pinecone.io/pricing/), [Datrabricks Data Intelligent Platform](https://www.databricks.com/product/data-intelligence-platform) or an [Anyscale Endpoint](https://www.anyscale.com/get-started) account and an associated key. Even better, install [OLlama](https://ollama.com/) on your laptop. 

All examples and notebooks are in Python, yet the concepts can be adapted to any programming language, so long as you can make REST API calls in the target language or use target language's SDK.

Some examples in this cook book are inspired(some modified, expanded or copied) from these resources, including:

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
 * Various medium blog posts cited as resources on chapter blogs published on medium.

 ## Directory Structure or Chapters for the nook

| Directory Names| Description | 
|---------------|-------------|
| genai_intro | General introduction to GenAI, Foundation Models, LLMS |
| agents| What are agents, agent archicture, why use them and how to write agents|
| assistants| Assistants extend LLM functionality to work and interact with external tools, enabling access to external services such as Web and datastores. How to work with OpenAI Assistants as agents |
| chatbots| Common application in customer service, how to write simple and conversational chatbots with LLMs, using prompting techniques|
| dspy| Quick overview of a declaractive framework to program LLMs: what and why and how to use DSPy|
|embeddings-and-vectordbs| Introduction to vector embeddings and how they play a role in semantic searches for LLM Gen apps. Vector embeddings are central to retrieval augmentation generation. Explore and use common vector stores as retreivers for indexed documents |
|evaluation| Evaluating LLM is not easy and messy; it can seem like a dark art. But some tool, like MLflow, provide exprimentation, logging, tracking and tracing to evaluate LLMs|
|fine-tuning | Common use of LLM to handle domain specific tasks is via fine-tuning. Why and when to fine-tune and how for your domain specific task, with your training data |
|function-calling| How to use both OpenAI and OSS LLM funciton calling to extend LLM application functionality |
| gpts| A walk through OpenAI's GPT models and how and when to use them and what for. How they compare with closed models.|
|llm-prompts| An introduction to myriad prompt engineering techniques using closed and open source LLM models|
|rags|Retrieval Augmentation Generation (RAG) is the TayLor Swift of LLM applications; everyone wants them; everyone writes about them; everyone builds them. An introduction to different types of RAGS, when to use them over fine-tunign, and how to to implement them for your data, increasing accuracy and decreasing halucinations in your responses|

 ## Current Blogs on this cookbook chapters

 * [Best Prompt Techniques for Best LLM Responses](https://medium.com/the-modern-scientist/best-prompt-techniques-for-best-llm-responses-24d2ff4f6bca)
* [LLM Beyond its Core Capabilities as AI Assistants or Agents](https://medium.com/@2twitme/llm-beyond-its-core-capabilities-as-ai-assistants-or-agents-704ffb972934)
* [Crafting Intelligent User Experiences: A Deep Dive into OpenAI Assistants API](https://medium.com/@2twitme/crafting-intelligent-user-experiences-a-deep-dive-into-openai-assistants-api-00439ace108a)
* [An Intuitive 101 Guide to Vector Embeddings](https://medium.com/@2twitme/an-intuitive-101-guide-to-vector-embeddings-ffde295c3558)
* [An Exploratory Tour of Retrieval Augmented Generation (RAG) Paradigm](https://medium.com/@2twitme/an-exploratory-tour-of-retrieval-augmented-generation-rag-paradigm-3940c1947d27)
* [An Exploratory Tour of DSPy: A Framework for Programing Language Models, not Prompting](https://medium.com/@2twitme/an-exploratory-tour-of-dspy-a-framework-for-programing-language-models-not-prompting-711bc4a56376)

## Environment files
Since I use either OpenAI, Anthropic, Google Gemmini, or Anyscale Endpoints, I provide respective environment template files: *env_anyscale_template*, *env_anthropic_template*, and *env_openai_template*. Add your keys and model name to either of the files and copy the file to **.env** to the top-level directory. To migrate any OpenAI code to Anyscale Endpoints, use this [simple guide](https://docs.endpoints.anyscale.com/guides/migrate-from-openai/). For the most part, the migration is seamless.

Also, you'll require some Python libraries. Use `pip install -r requirements.txt` to install them.

**Note**: Don't share your environment files with API keys publicly.

## Contributing
I welcome contributions. Let's make this a Generative AI developer community-driven resource. Your contributions can include additions or expansions of these how-to guides.

Check existing [issues](https://github.com/dmatrix/genai-cookbook/issues) and [pull requests](https://github.com/dmatrix/genai-cookbook/pulls) before contributing to avoid duplication. If you have suggestions for examples or guides, share them on the issues page.


Have GenAI fun! 🥳️
Jules
