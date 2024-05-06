import dspy
import argparse
from dspy_utils import TextCompletion, SummarizeText, \
    SummarizeTextAndExtractKeyTheme, TranslateText, \
    TextTransformationAndCorrection, TextCorrection, \
    TranslateTextToLanguage, GenerateJSON, \
    SimpleAndComplexReasoning, WordMathProblem

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
SUMMARY_THEME = """
    Isaac Newton sat under a tree when an apple fell, an event that, 
    according to popular legend, led to his contemplation of the forces
    of gravity. Although this story is often regarded as apocryphal or at 
    least exaggerated, it serves as a powerful symbol of Newton's insight 
    into the universal law that governs celestial and earthly bodies alike. 
    His formulation of the law of universal gravitation was revolutionary, 
    as it provided a mathematical explanation for both the motion of planets 
    and the phenomena observed on Earth. Newton's work in physics, captured 
    in his seminal work Philosophiæ Naturalis Principia Mathematica, laid the 
    groundwork for classical mechanics. His influence extended beyond his own 
    time, shaping the course of scientific inquiry for centuries to come.
"""

LANGUAGE_TEXT = """
    Welcome to New York for the United Nations General Council Meeting. 
    Today is a special day for us to celeberate all our achievments since 
    this global institute's formation.But more importantly, 
    we want to address how we can mitigate global conflict with 
    conversation and promote deterence, detente, and discussion.
    """

PIRATE_SPEAK = """
        Arrr matey! I be knowin' nuthin' 'bout them fancy words and grammatical rules. 
        Me and me heartie, we be chattin', and he don't be agreein' with me. 
        We ain't never gonna figure it out, I reckon. His scallywag of a dog 
        don't be listenin' well, always runnin' around and not comin' when ye call.
"""

INCORRECT_TEXT = """
    Yesterday, we was at the park, and them kids was playing. She don't like the way how they acted, but I don't got no problem with it. We seen a movie last night, and it was good, but my sister, she don't seen it yet. Them books on the shelf, they ain't interesting to me.
"""

LANGUAGE_TEXTS = ["""Bienvenidos a Nueva York para la Reunión del Consejo General de las Naciones Unidas. Hoy
es un día especial para celebrar todos nuestros logros desde la formación de este instituto global.
Pero más importante aún, queremos abordar cómo podemos mitigar el conflicto global con conversaciones
y promover la disuasión, la distensión y el diálogo.""",
            """Willkommen in New York zur Sitzung des Allgemeinen Rates der Vereinten Nationen. Heute
ist ein besonderer Tag für uns, um all unsere Errungenschaften seit der Gründung dieses globalen Instituts zu feiern.
Aber wichtiger ist, dass wir ansprechen möchten, wie wir globale Konflikte durch Gespräche mildern können
und Abschreckung, Entspannung und Diskussion fördern.""",
                  """Bienvenue à New York pour la réunion du Conseil Général des Nations Unies. Aujourd'hui,
c'est un jour spécial pour nous pour célébrer toutes nos réalisations depuis la formation de cette institution mondiale.
Mais plus important encore, nous voulons aborder comment nous pouvons atténuer les conflits mondiaux grâce à la conversation
et promouvoir la dissuasion, la détente et la discussion.""",
                  """欢迎来到纽约参加联合国大会议。今天对我们来说是一个特别的日子，我们将庆祝自该全球机构成立以来取得的所有成就。但更重要的是，我们想要讨论如何通过对话来缓解全球冲突，并促进遏制、缓和和讨论。
"""]

MATH_PROBLEM = """
    If my hourly rate is $117.79 per hour and I work 30 hours a week, 
    what is my yearly income?"
"""

if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser(description="Parse command line arguments.")

    # Add task argument
    parser.add_argument(
     "--task",
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Specify tasks to execute (default: all tasks).",
    )
    # Parse command line arguments
    args = parser.parse_args()

    # Setup OLlama environment on the local machine
    ollama_mistral = dspy.OllamaLocal(model='mistral',
                                      max_tokens=2500)
    dspy.settings.configure(lm=ollama_mistral)

    if 1 in args.task:
        # NLP Task 1: Text Generation and Completion
        # Use class signatures for text completion
        print("NLP Task 1: Text Generation and Completion")
        for prompt in PROMPTS:
            complete = dspy.Predict(TextCompletion)
            response = complete(in_text=prompt)
            print(f"{BOLD_BEGIN}Prompt: {prompt}{BOLD_END}")
            print(f"{BOLD_BEGIN}Completion: {response.out_text}{BOLD_END}")
            print("-------------------")

    if 2 in args.task:
        # NLP Task 2: Text Summarization
        # Use class signatures for summarization
        print("NLP Task 2: Text Summarization")
        summarize = dspy.Predict(SummarizeText)
        print("Summarization of text response:")
        response = summarize(text=SUMMARY)
        print(response.summary)
        print("-------------------")
    
    if 3 in args.task:
        # NLP Task 3: Text Summarization and Key Theme Extraction
        # Use class signatures for summarization and key theme extraction
        print("NLP Task 3: Text Summarization and Key Theme Extraction")
        summarize_theme = dspy.Predict(SummarizeTextAndExtractKeyTheme)
        print("Summary:")
        response = summarize_theme(text=SUMMARY_THEME)
        print(response.summary)
        print("Key Themes:")
        response.key_themes = response.key_themes.split("\n")
        print(response.key_themes)
        print("Takeaways:")
        print(response.takeaways)
        print("-------------------")

    if 4 in args.task:
        # NLP Task 4: Text Translation and Transliteration
        # Use class signatures for text translation
        print("NLP Task 4: Text Translation and Transliteration")
        translate = dspy.Predict(TranslateText)
        response = translate(text=LANGUAGE_TEXT)
        print(f"{BOLD_BEGIN}Language Text:{BOLD_END} {response.language}")
        print(f"{BOLD_BEGIN}Translated Text:{BOLD_END}")
        print(response.translated_text)
        print("-------------------")

    if 5 in args.task:
        # NLP Task 5: Text Transformation and Correction
        # Use class signatures for text transformation and correction
        print("NLP Task 5: Text Transformation and Correction")
        transform = dspy.Predict(TextTransformationAndCorrection)
        response = transform(text=PIRATE_SPEAK)
        print(f"{BOLD_BEGIN}Corrected Text:{BOLD_END}")
        print(response.corrected_text)
    
    if 6 in args.task:
        # NLP Task 6 Text Translation from Specific Language to English
        # Use class signatures for text translation
        print("NLP Task 6: Text Translation from Specific Language to English")
        translate = dspy.Predict(TranslateTextToLanguage)
        for text in LANGUAGE_TEXTS:
            response = translate(text=text)
            print(f"{BOLD_BEGIN}Language Text:{BOLD_END} {response.language}")
            print(f"{BOLD_BEGIN}Translated Text:{BOLD_END}")
            print(response.translated_text)
            print("-------------------")

    if 7 in args.task:
        # NLP Task 7: Text Correction for Grammatical Errors
        # Use class signatures for text correction
        print("NLP Task 6: Text Correction for Grammatical Errors")
        correct = dspy.Predict(TextCorrection)
        response = correct(text=INCORRECT_TEXT)
        print(f"{BOLD_BEGIN}Incorrect Text:{BOLD_END}")
        print(INCORRECT_TEXT)
        print(f"{BOLD_BEGIN}Corrected Text:{BOLD_END}")
        print(response.corrected_text)
        print("-------------------")

    if 8 in args.task:
        # NLP Task 8: Generate JSON Output
        # Use class signatures for JSON output generation
        print("NLP Task 8: Generate JSON Output")
        generate_json = dspy.Predict(GenerateJSON)
        response = generate_json()
        print(f"{BOLD_BEGIN}Generated JSON Output:{BOLD_END}")
        print(response.json_text)
        print("-------------------")

    if 9 in args.task:
        # NLP Task 9: Simple and Complex Reasoning
        # Use class signatures for simple and complex reasoning
        print("NLP Task 9: Simple and Complex Reasoning")
        reasoning = dspy.Predict(SimpleAndComplexReasoning)
        response = reasoning(numbers="'1', '2', '3', '4', '5', '7', '8', '11', '13', '17', '19', '23', '24', '29', '31', '37', '41', '43', '47', '53', '59', '61', '67', '71', '73', '79', '83', '89', '97'")
        print(f"{BOLD_BEGIN}Prime numbers:{BOLD_END} {response.prime_numbers}")
        print(f"{BOLD_BEGIN}Sum of Prime numbers:{BOLD_END} {response.sum_of_prime_numbers}")
        print(f"{BOLD_BEGIN}Sum is :{BOLD_END} {response.sum_is_even_or_odd }")
        print(f"{BOLD_BEGIN}Reasoning:{BOLD_END}")
        print(response.reasoning)
        print("-------------------")

    if 10 in args.task:
        # NLP Task 10: Word Math Problem
        # Use class signatures for word math problem
        print("NLP Task 10: Word Math Problem")
        word_math = dspy.Predict(WordMathProblem)
        response = word_math(problem=MATH_PROBLEM)
        print(f"{BOLD_BEGIN}Word Math Problem:{BOLD_END}")
        print(MATH_PROBLEM)
        print(f"{BOLD_BEGIN}Explanation:{BOLD_END}")
        print(response.explanation)
        print("-------------------")

