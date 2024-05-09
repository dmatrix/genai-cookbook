import dspy
import argparse
from dspy_utils import ChainOfThought, ChainOfThoughtTasks, \
    BOLD_BEGIN, BOLD_END


COT_TASKS_1 = """
        I'm offered $125.00 an hour contract job for six months.
        If I work 30 hours a week, how much will I make by the end of my contract.
"""
COT_TASKS_2 = """
At the recent holiday party, I got a coupon to join a health club
for wellness. If I joined before December 31, 2023 I get 35% discount on montly subscritpion fees
of $55.00 for one year, and the first three months' fees payments of $55.00 will be waived. 

The monthly payments for the health club subscription is $55.00

If I joined in January 2024, I get 25%, and only one month's fee is waived. 

Compute the best scenarios for saving costs for a one year subscription.
"""
COT_TASKS_3 = """
    Three girls, Emmy, Kasima, and Lina, had a fresh lemon juice booth stand
at the local community fair.

Emmy had 45 medium glasses of lemmon. She sold 43 glasses each at $1.25 per glass.

Kasima had 50 small glasses, and she sold all of them each at $1.15 per glass. 

And Lina had 25 large glasses and she sold only 11 glasses but at $1.75 per glass.

Of all the three girls, who made most money, and how many glasses each girl sold.
How many unsold glasses were left for each girl.

And finally, looking at all the numbers, which girl benefited most. That is, which
girl cleared her stock
"""

# Main function to execute CoT tasks
if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser(description="Parse command line arguments.")

    # Add task argument
    parser.add_argument(
     "--task",
        choices=[1, 2, 3, 4],
        type=int,
        nargs="+",
        default=[1, 2, 3,4],
        help="Specify tasks to execute (default: all tasks).",
    )
    # Parse command line arguments
    args = parser.parse_args()

    # Setup OLlama environment on the local machine
    ollama_mistral = dspy.OllamaLocal(model='mistral',
                                      max_tokens=2500)
    dspy.settings.configure(lm=ollama_mistral)

    # Execute tasks
    if 1 in args.task:
        # CoT Task 1: Solve the given text problem
        # Use class signatures ChainOfThought
        print("Chain of Thought Task 1.")
        cot = dspy.Predict(ChainOfThought)
        response = cot(problem_text=COT_TASKS_1)
        print(f"{BOLD_BEGIN}Problem:{BOLD_END} {COT_TASKS_1}")
        print(f"{BOLD_BEGIN}Result:{BOLD_END} {response.result}")
        print(f"{BOLD_BEGIN}Resaoning:{BOLD_END}: {response.reasoning}")
        print("-----------------------------\n")

    if 2 in args.task:
        # CoT Task 2: Solve the given text problem
        # Use class signatures ChainOfThought
        print("Chain of Thought Task 2.")
        cot = dspy.Predict(ChainOfThought)
        response = cot(problem_text=COT_TASKS_2)
        print(f"{BOLD_BEGIN}Problem:{BOLD_END} {COT_TASKS_2}")
        print(f"{BOLD_BEGIN}Result:{BOLD_END} {response.result}")
        print(f"{BOLD_BEGIN}Resaoning:{BOLD_END}: {response.reasoning}")
        print("-----------------------------\n")

    if 3 in args.task:
        # CoT Task 3: Solve the given text problem
        # Use class signatures ChainOfThought
        print("Chain of Thought Task 3.")
        cot = dspy.Predict(ChainOfThought)
        response = cot(problem_text=COT_TASKS_3)
        print(f"{BOLD_BEGIN}Problem:{BOLD_END} {COT_TASKS_3}")
        print(f"{BOLD_BEGIN}Result:{BOLD_END} {response.result}")
        print(f"{BOLD_BEGIN}Resaoning:{BOLD_END}: {response.reasoning}")
        print("-----------------------------\n")

