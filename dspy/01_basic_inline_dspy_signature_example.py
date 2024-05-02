import dspy

QUESTION = ["What is dark matter in the universe?",
            "Why did the dinosaurs go extinct?",
            "why did the chicken cross the road?"
]

SENTIMENTS = ["Movie was great!", 
              "Movie was terrible!",
              "Movie was okay!"]

SUMMARY = """
    Writing signatures is far more modular, adaptive, and 
    reproducible than hacking at prompts or finetunes. 
    The DSPy compiler will figure out how to build a 
    highly-optimized prompt for your LM (or finetune your 
    small LM) for your signature, on your data, and within 
    your pipeline. 

    In many cases, we found that compiling leads to better prompts 
    than humans write. Not because DSPy optimizers are more creative 
    than humans, but simply because they can try more things and 
    tune the metrics directly.
"""

if __name__ == "__main__":

    # Setup Ollama environment
    ollama_mistral = dspy.OllamaLocal(model='mistral')
    dspy.settings.configure(lm=ollama_mistral)

    # Use inline signatgure for question answering
    for question in QUESTION:
        answer = dspy.Predict('question -> answer')
        print(f"Question: {question}")
        print(f"Answer: {answer(question=question).answer}")
        print("-------------------")

    # Use line signatures for classification
    for sentiment in SENTIMENTS:
        classify = dspy.Predict('sentence -> sentiment')
        print(f"{classify(sentence=sentiment).sentiment}")
        print("-------------------")

    # use line signatures for summarization
    summarize = dspy.Predict('text -> summary')
    print("Summary:")
    print(summarize(text=SUMMARY).summary)