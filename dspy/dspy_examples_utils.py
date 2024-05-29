import dspy
from dspy.datasets import HotPotQA
import warnings

# Few-shot examples for the Program of Thought Module

FEW_SHOT_EXAMPLES = [
        {"question": """What is the elevation range for the area that the eastern sector of the
                    Colorado orogeny extends into?""",
        "thought_1": """I need to search Colorado orogeny, find the area that the eastern sector
                        of the Colorado orogeny extends into, then find the elevation range of the
                        area.""",
        "action_1":"""Search[Colorado orogeny]""",
        "observation_1":"""The Colorado orogeny was an episode of mountain building (an orogeny) in
                          Colorado and surrounding areas. It occurred during the Late Cretaceous and early Paleogene periods of the""",
        "thought_2": """It does not mention the eastern sector. So I need to look up eastern sector.""",
        "action_2":"""Search[eastern sector]""",
        "observation_2":"""(Result 1 / 1) The eastern sector extends into the High Plains and is called
                        the Central Plains orogeny. .""",
        "thought_3": """The eastern sector of Colorado orogeny extends into the High Plains. So I
                    need to search High Plains and find its elevation range.""",
        "action_3":"""Search[High Plains]""",
        "observation_3":"""High Plains refers to one of two distinct land regions""",
        "thought_4": """I need to instead search High Plains (United States).""",
        "action_4":"""Search[High Plains (United States)]""",
        "observation_4":"""The High Plains are a subregion of the Great Plains. From east to west, the
                        High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130
                    m).[3]""",
        "thought_5": """ High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer
                    is 1,800 to 7,000 ft.""",
        "answer":"""Finish[1,800 to 7,000 ft]"""
     },
    {   "question": """Based on information provided to you upto 2023, Who was Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?""",
        "thought_1": """I need to recall information on Olivia Wilde's relationship status as of my latest update and identify who her boyfriend was at that time. Then, I need to determine his age and calculate his age raised to the 0.23 power.""",
        "action_1":"""Recall[Olivia Wilde's boyfriend as of latest update]""",
        "observation_1":"""As of my last update, Olivia Wilde was known to be in a relationship with Harry Styles.""",
        "thought_2": """Now that I have identified Harry Styles as Olivia Wilde's boyfriend, I need to recall Harry Styles' birth date to calculate his current age.""",
        "action_2":"""Recall[Harry Styles' birth date]""",
        "observation_2":"""Harry Styles was born on February 1, 1994.""",
        "thought_3": """To determine Harry Styles' current age, I will calculate the number of years from his birth year (1994) to the current year, which would be 2023.""",
        "action_3":"Calculate[2023 - 1994]",
        "observation_3":"The calculation yields 29 years (2023 - 1994 = 29).",
        "thought_4": "Now I have Harry Styles' current age, I need to raise it to the power of 0.23 as per the original question.",
        "action_4":"Calculate[29^0.23]",
        "observation_4":"Raising 29 to the power of 0.23 results in approximately 2.66.",
        "thought_5": "The result of Harry Styles' current age raised to the power of 0.23 is approximately 2.66",
        "answer":"Finish[2.66]",
     },
     {   "question": """Based on information provided to you upto 2023, how do you calculate the value
                    of PI. With its current value of PI, given a circle of diameter 2 meters, what its its 
                    circumcernce and area?""",
        "thought_1": """To calculate the value of pi (π), mathematicians use various algorithms and numerical 
                        methods because π is an irrational number and cannot be expressed as a simple fraction. 
                        Historically, π has been approximated by methods like polygon approximation or series expansions.""",
        "action_1":""" No action required here because the value of π is already known and accepted as approximately 3.14159.""",
        "observation_1":"The value of π is widely known to be approximately 3.14159.",
        "thought_2": """To find the circumference of a circle, the formula is C = πd, where d is the diameter, 
                        and π is approximated as 3.14159. To find the area of a circle, 
                        the formula is A = πr^2, where r is the radius (half of the diameter).""",
        "action_2": """Calculate circumference and area using the known value of π.""",
        "observation_2":"""For a circle with a diameter of 2 meters, the radius would be 1 meter (half of the diameter).
                            Circumference calculation:
                            C = πd = π * 2 = 3.14159 * 2 ≈ 6.28318 meters
                            Area calculation:
                            A = πr^2 = π * 1^2 = 3.14159 * 1 * 1 ≈ 3.14159 square meters
                        """,
        "thought_3": """The circumference of a circle with a diameter of 2 meters is approximately 6.28318 meters, 
                        and the area is approximately 3.14159 square meters.""", 
        "answer":"Finish[Circumference: 6.28318 meters, Area: 3.14159 square meters]", 
     }
]

def get_few_shot_dspy_examples(debug=False):
    examples = []
    for example in FEW_SHOT_EXAMPLES:
        dspy_example = dspy.Example(**example).with_inputs('question')
        if debug:
            print(dspy_example)
        examples.append(dspy_example)

    return examples

if __name__ == "__main__":
   
   warnings.filterwarnings("ignore")
   examples = get_few_shot_dspy_examples()
   # Display the few-shot examples for training
   for idx, example in enumerate(examples):
    dspy_example = dspy.Example(**example)
    print(f"DSPy Example :{idx + 1}")
    print(f"question     : {dspy_example.question}")
    print(f"thought_1    : {dspy_example.thought_1}")
    print(f"observation_1: {dspy_example.observation_1}")
    print(f"action_1     : {dspy_example.action_1}")
    print(f"answer       : {dspy_example.answer}")
    print("---\n")