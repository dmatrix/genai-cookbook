import dspy
from dspy_utils import QuestionAnswer, ClassifyEmotion, SummarizeText

QUESTIONS = ["What is dark matter in the universe?",
            "Why did the dinosaurs go extinct?",
            "why did the chicken cross the road?"
]

SENTIMENTS = [
        "This movie is a true cinematic gem, blending an engaging plot with superb performances and stunning visuals. A masterpiece that leaves a lasting impression.",
         "Regrettably, the film failed to live up to expectations, with a convoluted storyline, lackluster acting, and uninspiring cinematography. A disappointment overall.",
         "The movie had its moments, offering a decent storyline and average performances. While not groundbreaking, it provided an enjoyable viewing experience.",
         "This city is a vibrant tapestry of culture, with friendly locals, historic landmarks, and a lively atmosphere. An ideal destination for cultural exploration.",
         "The city's charm is overshadowed by traffic congestion, high pollution levels, and a lack of cleanliness. Not recommended for a peaceful retreat.",
         "The city offers a mix of experiences, from bustling markets to serene parks. An interesting but not extraordinary destination for exploration.",
         "This song is a musical masterpiece, enchanting listeners with its soulful lyrics, mesmerizing melody, and exceptional vocals. A timeless classic.",
         "The song fails to impress, featuring uninspiring lyrics, a forgettable melody, and lackluster vocals. It lacks the creativity to leave a lasting impact.",
         "The song is decent, with a catchy tune and average lyrics. While enjoyable, it doesn't stand out in the vast landscape of music.",
         "A delightful cinematic experience that seamlessly weaves together a compelling narrative, strong character development, and breathtaking visuals.",
         "This film, unfortunately, falls short with a disjointed plot, subpar performances, and a lack of coherence. A disappointing viewing experience.",
         "While not groundbreaking, the movie offers a decent storyline and competent performances, providing an overall satisfactory viewing experience.",
         "This city is a haven for culture enthusiasts, boasting historical landmarks, a rich culinary scene, and a welcoming community. A must-visit destination.",
         "The city's appeal is tarnished by overcrowded streets, noise pollution, and a lack of urban planning. Not recommended for a tranquil getaway.",
         "The city offers a diverse range of experiences, from bustling markets to serene parks. An intriguing destination for those seeking a mix of urban and natural landscapes.",
         "Mr. Xxx Yyy Zzz was curious and dubious"
]

SUMMARY = """
    DSPy is a framework for algorithmically optimizing LM prompts and weights, 
    especially when LMs are used one or more times within a pipeline. To use 
    LMs to build a complex system without DSPy, you generally have 
    to: (1) break the problem down into steps, (2) prompt your 
    LM well until each step works well in isolation, (3) tweak 
    the steps to work well together, (4) generate synthetic 
    examples to tune each step, and (5) use these examples to finetune 
    smaller LMs to cut costs. Currently, this is hard and messy: every 
    time you change your pipeline, your LM, or your data, all prompts 
    (or finetuning steps) may need to change.

    To make this more systematic and much more powerful, DSPy does two things. 
    First, it separates the flow of your program (modules) from the 
    parameters (LM prompts and weights) of each step. 
    Second, DSPy introduces new optimizers, which are LM-driven 
    algorithms that can tune the prompts and/or the weights of 
    your LM calls, given a metric you want to maximize.

    DSPy can routinely teach powerful models like GPT-3.5 or GPT-4 and 
    local models like T5-base or Llama2-13b to be much more reliable at 
    tasks, i.e. having higher quality and/or avoiding specific failure patterns. 
    DSPy optimizers will "compile" the same program into different instructions, 
    few-shot prompts, and/or weight updates (finetunes) for each LM. 
    This is a new paradigm in which LMs and their prompts fade into the background 
    as optimizable pieces of a larger system that can learn from data. tldr; 
    less prompting, higher scores, and a more systematic approach to solving hard 
    tasks with LMs.
"""

if __name__ == "__main__":

    # Setup Ollama environment on the local machine
    ollama_mistral = dspy.OllamaLocal(model='mistral')
    dspy.settings.configure(lm=ollama_mistral)

    for question in QUESTIONS:
        answer = dspy.Predict(QuestionAnswer)
        print(f"Question: {question}")
        print(answer(question=question))
        print("-------------------")

    for sentence in SENTIMENTS:
        classify = dspy.Predict(ClassifyEmotion)
        print(f"sentiment: {sentence}")
        print(classify(sentence=sentence))
        print("-------------------")

    # Use class signatures for summarization
    summarize = dspy.Predict(SummarizeText)
    print("Summary:")
    print(summarize(text=SUMMARY))
