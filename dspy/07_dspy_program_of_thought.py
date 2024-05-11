import dspy
import warnings
import argparse
from dspy_utils import POT, BOLD_BEGIN, BOLD_END

POT_TASKS_1 = """
            Sarah has 5 apples. She buys 7 more apples from the store. 
            How many apples does Sarah have now?"""

POT_TASKS_2 = """
            What is the area of a triangle if its base is 4ft and height
            is 7ft?.
    """
POT_TASKS_3 = """
            How to write a Python function to check a prime nunber.
    """

# Main function to execute CoT tasks
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Create argument parser
    parser = argparse.ArgumentParser(description="Parse command line arguments.")

    # Add task argument
    parser.add_argument(
     "--task",
        choices=[1, 2, 3],
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Specify tasks to execute (default: all tasks).",
    )
    # Parse command line arguments
    args = parser.parse_args()

    # Setup OLlama environment on the local machine
    ollama_mistral = dspy.OllamaLocal(model='mistral',
                                      max_tokens=3500)
    dspy.settings.configure(lm=ollama_mistral)

    # Execute POT tasks
    if 1 in args.task:
        # POT Task 1: Solve the given text problem
        # Use class Module POT
        print("Program of Thought Task 1.")
        pot = POT()
        response = pot(question=POT_TASKS_1)
        print(f"{BOLD_BEGIN}Question:{BOLD_END} {POT_TASKS_1}")
        if hasattr(response, "answer"):
            print(f"{BOLD_BEGIN}Answer  :{BOLD_END} {response.answer}")
        print("-----------------------------\n")

    if 2 in args.task:
        # POT Task 2: Solve the given text problem
        # Use class Module POT
        print("Program of Thought Task 2.")
        pot = POT()
        response = pot(question=POT_TASKS_2)
        print(f"{BOLD_BEGIN}Question:{BOLD_END} {POT_TASKS_2}")
        if hasattr(response, "answer"):
            print(f"{BOLD_BEGIN}Answer  :{BOLD_END} {response.answer}")
        print("-----------------------------\n")

    if 3 in args.task:      
        # POT Task 3: Solve the given text problem
        # Use class Module POT
        print("Program of Thought Task 3.")
        pot = POT()
        response = pot(question=POT_TASKS_3)
        print(f"{BOLD_BEGIN}Question:{BOLD_END} {POT_TASKS_3}")
        if hasattr(response, "answer"):
            print(f"{BOLD_BEGIN}Answer  :{BOLD_END} {response.answer}")
        print("-----------------------------\n")
