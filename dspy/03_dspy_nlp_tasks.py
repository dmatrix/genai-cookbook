import dspy
from dspy_utils import TextCompletion, SummarizeText

BOLD_BEGIN = "\033[1m"
BOLD_END = "\033[0m"

PROMPTS = [
        "On cold winter nights, the wolves in Siberia ...",
        "On the day Franklin Benjamin realized his passion for printer, ...",
        "The ancient city of Atlantis was known for its ...",
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

    # Setup OLlama environment on the local machine
    ollama_mistral = dspy.OllamaLocal(model='mistral',
                                      max_tokens=1000)
    dspy.settings.configure(lm=ollama_mistral)

    # NLP Task 1: Text Generation and Completion
    # Use class signatures for text completion
    for prompt in PROMPTS:
        complete = dspy.Predict(TextCompletion)
        response = complete(in_text=prompt)
        print(f"{BOLD_BEGIN}Prompt: {prompt}{BOLD_END}")
        print(f"{BOLD_BEGIN}Completion: {response.out_text}{BOLD_END}")
        print("-------------------")

    # NLP Task 2: Text Summarization
    # Use class signatures for summarization
    summarize = dspy.Predict(SummarizeText)
    print("Summary:")
    print(summarize(text=SUMMARY))
    print("-------------------")
