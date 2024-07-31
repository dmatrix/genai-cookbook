# GenAI Cookbook

Generative Artificial Intelligence (GenAI) is rapidly transforming various domains by introducing unprecedented levels of novelty, creativity, and productivity, "creating the largest technological shift in our lifetimes." With its ability to respond to human-like natural language instructions, GenAI adds a new interactive dimension to AI applications. It generates new content—text, audio, images, and video—from existing data, achieving feats previously unimaginable. Also, GenAI excels in productivity, outperforming humans in some tasks in speed and efficiency.

<a href="https://github.com/dmatrix/genai-cookbook"><img src="https://img.shields.io/github/stars/dmatrix/genai-cookbook" alt="github-stars"></a>
<a href="https://twitter.com/2twitme"><img src="https://img.shields.io/twitter/follow/2twitme?label=Follow" alt="twitter"></a>

<img src=images/gen_ai_cookbook_img_1.png>

Welcome to the GenAI cookbook, a blend of how-to recipes and guides for using LLMs. This collection covers a wide range of natural language processing (NLP) applications using language models:

**LLM Prompting Strategies**: Get-started examples to prompting techniques 
**DSPy Framework**: Explore alternatives to prompt engineering with declarative and programmatic approaches to programming LLMs
**Simple Chatbots**: Learn the basics of building interactive AI
**Retrieval Generation Augmentation (RAGs)**: Incorporate personal or organizational data for more accurate responses
**Fine-Tuning LLMs**: Customize models for domain-specific tasks for form, format, and tonality
**Extending LLM Functionality**:  Use agents and functions to enhance capabilities by interacting with external tools and data sources
**Employing Leading Language Models**: Practical guides for using models and inference platforms from OpenAI, Anthropic, Gemini, Meta, Anyscale Endpoints, OLlama, and Databricks.

Aimed at beginner developers, this book provides example code and how-to guides, showcasing how to use APIs and frameworks from leading platforms. Start your journey and discover the potential of GenAI.

To try these examples, you'll need an [OpenAI](https://platform.openai.com/docs/introduction) account and an associated API key, [Anthropic](https://docs.anthropic.com/claude/docs/intro-to-claude), [Pinecone](https://www.pinecone.io/pricing/), [Datrabricks Data Intelligent Platform](https://www.databricks.com/product/data-intelligence-platform) or an [Anyscale Endpoint](https://www.anyscale.com/get-started) account and an associated key. Even better, install [OLlama](https://ollama.com/) on your laptop. 

All examples and notebooks are in Python, yet the concepts can be adapted to any programming language, so long as you can make REST API calls in the target language or use target language's SDK.

Some examples in this cookbook are inspired (some modified, expanded or copied) from these resources, including:

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
| genai_intro | General introduction to GenAI, Foundation Models, GPTs |
| agents| What are agents, evolving agent archicture, why use them and how to write agents|
| assistants| OpenAI Assistants extend LLM functionality to work and interact with external tools, enabling access to external services such as Web services and datastores. How to work with OpenAI Assistants to implement agents |
| chatbots| Common application in customer service, how to write simple and conversational chatbots with LLMs, using prompting techniques|
| dspy| Quick overview of a declaractive framework to program LLMs: what and why and how to use DSPy|
|embeddings-and-vectordbs| Introduction to vector embeddings and their a role in semantic searches for LLM Gen apps. Vector embeddings are central to retrieval augmentation generation. Explore common vector stores as retreivers for indexed documents |
|evaluation| Evaluating LLM is not easy and messy;it may seem like a dark art. But some tools, like MLflow, provide experimentation, logging, tracking and tracing to evaluate LLMs|
|fine-tuning | Common use of LLM to handle domain specific tasks is via fine-tuning. Why and when to fine-tune for your domain specific task to customized responses, tone, and format|
|function-calling| How to use both OpenAI and open-source LLM funciton calling to extend LLM functionality |
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
Since I use either OpenAI, Anthropic, Google Gemmini, or Anyscale Endpoints, I provide respective environment template files: *env_anyscale_template*, *env_anthropic_template*, *env_google_template*, and *env_openai_template*. 

Add your keys and model name to either of the files and copy the file to **.env** to the top-level directory. To migrate any OpenAI code to Anyscale Endpoints, use this [simple guide](https://docs.endpoints.anyscale.com/guides/migrate-from-openai/). For the most part, the migration is seamless.

Also, you'll require some Python libraries. Use `pip install -r requirements.txt` to install them.

**Note**: Don't share your environment files with API keys publicly.

## Contributing
I welcome contributions. Let's make this a Generative AI developer community-driven resource. Your contributions can include additions or expansions of these how-to guides.

Check existing [issues](https://github.com/dmatrix/genai-cookbook/issues) and [pull requests](https://github.com/dmatrix/genai-cookbook/pulls) before contributing to avoid duplication. If you have suggestions for examples or guides, share them on the issues page.


Have GenAI fun! 🥳️

Jules
