import dspy
import warnings
from dspy_utils import RAG, BOLD_BEGIN, BOLD_END, \
        ZeroShotEntityNameRecognition, DialogueGeneration


USER_TEXTS = [
    "Tesla, headquartered in Palo Alto, was founded by Elon Musk. The company recently announced a collaboration with NASA to explore sustainable technologies for space travel.",
    "The United States of America is a country in North America. It is the third largest country by total area and population. The capital is Washington, D.C., and the most populous city is New York City. The current president is Joe Biden. The country was founded on July 4, 1776, and declared independence from Great Britain. And its founding fathers are George Washington, Thomas Jefferson, and Benjamin Franklin.",
]

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print(BOLD_BEGIN + "Zero Short Learning with DSPy" + BOLD_END)

     # Setup OLlama environment on the local machine
    ollama_mistral = dspy.OllamaLocal(model='mistral',
                                      max_tokens=2500)
    dspy.settings.configure(lm=ollama_mistral)

    # Create our Zero Short Signature instance
    zero = dspy.Predict(ZeroShotEntityNameRecognition, max_iters=5)

    # Create our user text
    user_text = """Tesla, headquartered in Palo Alto, was founded by Elon Musk. The company recently announced a collaboration with NASA to explore sustainable technologies for space travel.
            """ 
    # Run the Zero Short
    for user_text in USER_TEXTS:
        result = zero(text=user_text)
        print(f"{BOLD_BEGIN} User Input: {user_text} {BOLD_END}")
        print(f"{BOLD_BEGIN} Named Entities: {result.entities} {BOLD_END}")
        print("--------------------------")

    # Create an instance of DialogueGeneration
    dialog = dspy.Predict(DialogueGeneration, max_iters=5)
    # Run the DialogueGeneration
    problem_text = """Hello, I've been experiencing issues with the software. It keeps crashing whenever I try to open a specific file. 
Can you help?"""
    result = dialog(problem_text=problem_text)
    print(f"{BOLD_BEGIN} User Input {BOLD_END}: {problem_text}")
    print(f"{BOLD_BEGIN} Dialogue {BOLD_END}: {result.dialogue}")
    print("--------------------------")
    # Inspect the prompt history 
    print(f"{BOLD_BEGIN} Prompt History {BOLD_END}:")
    print(ollama_mistral.inspect_history(n=1))
    print("--------------------------")
