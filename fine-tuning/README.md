## Zero-shot prompting

Zero-prompt learning is a challenging yet fascinating area where models are trained to perform tasks without explicit learning examples in the input prompt. Here are some notable examples that can be used with GPT-3 and Llama Language Models.

GPT-3, Llama 2, and Claude are powerful language models. The have demonstrated zero-shot learning. That is, without specific learning prompts or examples, it can generate coherent and contextually relevant responses, showcasing its ability to understand and respond to diverse queries.

### Image Classification:

In zero-shot image classification, a model can classify images into categories it has never seen during training. This is achieved without providing explicit examples of those new categories, showcasing the model's ability to generalize.

### Text Summarization:

Models trained for zero-prompt text summarization can generate concise and meaningful summaries without receiving explicit instructions. The model learns to distill key information from the input text.

### Anomaly Detection:

In zero-shot anomaly detection, models can identify unusual patterns or outliers in data without being explicitly trained on examples of anomalies. This is particularly useful in cybersecurity and fraud detection.

### Language Translation:

Zero-shot learning in language translation allows models to translate languages they haven't seen during training. The model learns to generalize its translation capabilities without specific examples for each language pair.

### Named Entity Recognition (NER):

Models trained with zero-prompt learning for NER can identify and categorize named entities in text without being explicitly provided with examples for each specific entity.

### Dialogue Generation:

Zero-shot dialogue generation models can engage in conversations and respond appropriately to user input without being given explicit dialogues as training examples.

### Speech Recognition:

Models trained for zero-shot speech recognition can accurately transcribe spoken words even in languages or accents not explicitly included in the training data.

These examples highlight the remarkable capacity of models to generalize and adapt to diverse tasks or data distributions without relying on explicit prompts or examples during training. 

Note: Except for image classification and speech recognition, which require
GPT-4 or multi-model LLMs, all notebooks in `llm-prompts` are examples of zero-shot learning. To extend those examples, We have added two xadditional how-to guides for NER and Dialogue generation: [zero_shot_prompting](./1_zero_shot_prompting.ipynb)

## Few-shot prompting

Few-shot learning is a machine learning paradigm where a model is trained to perform a task with very few examples or prompts. Unlike traditional machine learning that often requires large amounts of labeled data, few-shot learning enables models to generalize and adapt to new tasks with minimal examples. This is particularly useful in scenarios where obtaining extensive labeled datasets is challenging or impractical.

Few-shot prompting is akin to an early fine-tuning stage for training models on specific, small datasets or formats. It involves incorporating limited examples as part of the prompt context, encompassing both zero-shot and few-shot prompting in this fine-tuning section.

### When to Use Few-shot Learning:

There are few examples where few-shot learning is applicable. 

#### Rapid Adaptation:

When you need a model to quickly adapt to new tasks or domains with minimal training data, few-shot learning allows for rapid learning and deployment.


#### Limited Labeled Data:

Few-shot learning is beneficial when you have a limited amount of labeled data for a specific task, making it challenging to train a traditional model effectively.

Let's explore couple of examples of few-short prompt learning in our our notebook: [2_few_shot_prompting](./2_few_shot_prompting.ipynb)