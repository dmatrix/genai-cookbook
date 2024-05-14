import os
import sys
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
            How to write a Python function to check a prime nunber?
    """
POT_TASKS_4 = """
            How to write a Python function to generate fibonacci series?   
"""
POT_TASKS_5 = """
            How to write Python code to find three smallest prime numbers greater than 30,000?
"""

POT_TASK_6 = """
 How to write a Python code, using fractions python module, that generates fractions between 1 and 10?
"""
POT_TASK_7 = """
Given the following SQL schema for tables
Table clicks, columns = [target_url, orig_url, user_id, clicks]
Table users, columns = [user_id, f_name, l_name, e_mail, company, title], generate
an SQL query that computes in the descening order of all the clicks. Also, for
each user_id, list the f_name, l_name, company, and title

"""

POT_TASKS = [POT_TASKS_1, POT_TASKS_2, POT_TASKS_3, 
             POT_TASKS_4, POT_TASKS_5, POT_TASK_6, 
             POT_TASK_7]

# Utility function to execute PoT tasks
def run_pot_task(task, question=None, args=None, history=None):
    print(f"Program of Thought Task {task}.")
    print(f"{BOLD_BEGIN}Question:{BOLD_END}{question}")
    
    pot = POT()
    response = pot(question=question)
    
    if hasattr(response, "answer"):
        print(f"{BOLD_BEGIN}Answer  :{BOLD_END}{response.answer}")
    
    print("-----------------------------\n")
    
    if history:
        print(f"{BOLD_BEGIN}Prompt History:{BOLD_END}") 
        print(ollama_mistral.inspect_history(n=history))
        print("===========================\n")


# Main function to execute CoT tasks
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Create argument parser
    parser = argparse.ArgumentParser(description="Parse command line arguments.")

    # Add task argument
    parser.add_argument(
     "--task",
        choices=[1, 2, 3, 4, 5, 6, 7],
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7],
        help="Specify tasks to execute (default: all tasks).",
    )
    parser.add_argument("--history", 
                        choices=[1, 2, 3], 
                        type=int, 
                        default=None,
                        help="Specify a history value (1, 2, or 3)")

    # Parse command line arguments
    args = parser.parse_args()

    # Setup OLlama environment on the local machine
    ollama_mistral = dspy.OllamaLocal(model='mistral',
                                      max_tokens=3500)
    dspy.settings.configure(lm=ollama_mistral)

    # Execute POT tasks
    if args.task:
       for task in args.task:
          run_pot_task(task, question=POT_TASKS[task-1], args=args, history=args.history)
    else: 
        raise ValueError("Invalid task number. Please specify a valid task number.")
