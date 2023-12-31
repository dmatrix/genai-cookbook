# GenAI Cookbook

<img src=images/gen_ai_cookbook_img_1.png>

Welcome to a mixture of Gen AI cookbook **how-to** recipes for Gen AI applications. These simple guides span across get-started examples on LLM prompting strategies, building simple chatbots, retrieval generation augmentation (RAGs), LLM-based applications on ChatGPT or Anyscale Endpoints, etc. 

You'll find examples code and/or guides for common tasks using the OpenAI API and [Ray framework](https://www.ray.io/). 

To try these examples, you'll need a free [OpenAI](https://platform.openai.com/docs/introduction) account and an associated API key or an [Anyscale Endpoint](https://www.anyscale.com/get-started) account and an associated key. 

All examples are in Python, yet the concepts can be adapted to any programming language, so long as you can make REST API calls in the target language or use target language's SDK.

The examples in this cook are inspired (some modified, expanded or copied) from these resources, including:

 * [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
 * [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
 * [Prompt Engineering Guide](https://www.promptingguide.ai/introduction) and [Prompt Engineering course](https://maven.com/dair-ai/prompt-engineering-llms?promoCode=MAVENMONDAY) by Elvis Saravia
 * [How I Won Singapore's GPT-4 Prompt Engineering Competition](https://towardsdatascience.com/how-i-won-singapores-gpt-4-prompt-engineering-competition-34c195a93d41) by Sheila Teo
 * [Ray Documentation](https://docs.ray.io/en/latest/) and [Anyscale blogs](https://www.anyscale.com/blog)
 * [Anyscale Endpoints Documentation](https://docs.endpoints.anyscale.com/)
 * [ChatGPT Prompt Engineering for Developers](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction)
 * [LangChain for LLM Application Development](https://learn.deeplearning.ai/langchain/lesson/1/introduction)
 * [Building Systems with the ChatGPT API](https://learn.deeplearning.ai/chatgpt-building-system/lesson/1/introduction)
 * Various blog posts on medium.

## Environment files
Since I use use either OpenAI or Anyscale Endpoints, I provide two
environment template files: **env_anyscale_template** and **env_openai_template**. Add your keys and model name to either of the files and copy the file to **.env** to the top-level directory. To migrate any OpenAI code to Anyscale Endpoints, use this [simple guide](https://docs.endpoints.anyscale.com/guides/migrate-from-openai/). For the most part, the migration is seamless.

Also, you'll require some Python libraries. Use `pip install -r requirements.txt` to install them.

**Note**: Don't share your environment files with API keys publicly.

## Contributing
I welcome contributions. Let's make this a Generative AI developer community-driven resource. Your contributions can include additions or expansions of these how-to guides.

Check existing [issues](https://github.com/dmatrix/genai-cookbook/issues) and [pull requests](https://github.com/dmatrix/genai-cookbook/pulls) before contributing to avoid duplication. If you have suggestions for examples or guides, share them on the issues page.


Have GenAI fun! 🥳️

Jules
