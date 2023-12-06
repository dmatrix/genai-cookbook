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
GPT-4 or multi-model LLMs, all notebooks in `llm-prompts` are examples of zero-shot learning. To extend those examples, We have added 
two added two additional how-to guides for NER and Dialogue generation: [zero_shot_prompting](./1_zero_shot_prompting.ipynb)