import dspy
from typing import List

BOLD_BEGIN = "\033[1m"
BOLD_END = "\033[0m"

class QuestionAnswer(dspy.Signature):
    """Answer questions based on the input question."""

    question = dspy.InputField()
    answer   = dspy.OutputField()

class ClassifyEmotion(dspy.Signature):
    """Classify emotion based on the input sentence
        and provide the sentiment as the output."""
    sentence  = dspy.InputField()
    sentiment = dspy.OutputField(desc="generate reponse as positive, negative or neutral")

class SummarizeText(dspy.Signature):
    """Summarize a given text into a succinct summary, in no more
    than a combination of 10 or 15 simple, compound, complex, and 
    compound-complex complete sentences. Don't truncate the text.
    """
    text    = dspy.InputField()
    summary = dspy.OutputField()

class SummarizeTextAndExtractKeyTheme(dspy.Signature):
    """Summarize a given text succinctly, in no more
    than a combination of six simple, compound, complex, and 
    compound-complex sentences. Extract the key themes 
    from the text, label it as 'Key Subjects:', and 
    enumerate 'Takeaways:'.
    """
    text       = dspy.InputField()
    summary    = dspy.OutputField()
    key_themes = dspy.OutputField()
    takeaways  = dspy.OutputField()

class TranslateText(dspy.Signature):
    """Given text indentify the language, translate into 
        Spanish, French, German, Portugese, Japense, Korean, 
        and Mandarin."""

    text      = dspy.InputField()
    language  = dspy.OutputField()
    translated_text = dspy.OutputField()

class TranslateTextToLanguage(dspy.Signature):
    """Given text, indentify the language and translate into 
    English language."""

    text      = dspy.InputField()
    language  = dspy.OutputField()
    translated_text = dspy.OutputField()


class TextCompletion(dspy.Signature):
    """Complete a given text with more words to the best 
    of your acquired knowledge. Don't truncate the generated
    response.
    """
    
    in_text  = dspy.InputField()
    out_text = dspy.OutputField()

class TextTransformationAndCorrection(dspy.Signature):
    """Transform the given text on Pirate speak to a standard english text. 
    Correct any grammatical errors. Provide the corrected 
    text as the output.
    """

    text = dspy.InputField()
    corrected_text = dspy.OutputField()

class TextCorrection(dspy.Signature):
    """Correct the given text for any grammatical errors. 
    Provide the corrected text as the output.
    """

    text = dspy.InputField()
    corrected_text = dspy.OutputField()

class GenerateJSON(dspy.Signature):
    """Generate five distinct products on training shoes. 
       Generate products and format them all in a single JSON object.
       For each product, the JSON object should 
       contain items: Brand, Description, Size, Gender (Male, Female or Unisex), 
       Price, and Review (three customer reviews.
    """

    json_text = dspy.OutputField(desc='key-value pairs')

class TextCategorizationAndSentimentAnalsysis(dspy.Signature):
    """Categorize the given text into one of the following categories:
    Technical support, Billing, Account Management, New Customer or General inquiry.
    If you can't classify the text, default to 'General inquiry.'
    If customer text contatins a foul language, then respond with 
    'No need for foul language. Please be respectful."
    Also provide the sentiment of the text as the output as positive, negative or neutral.
    """

    text      = dspy.InputField()
    category  = dspy.OutputField()
    sentiment = dspy.OutputField()

class SimpleAndComplexReasoning(dspy.Signature):
    """Given a list of numbers identify all prime numbers, 
    output the list of all prime numbers, add the list and identify
    if the sum is even or odd. 
    """
    numbers = dspy.InputField(format=list)
    prime_numbers = dspy.OutputField(desc="no need for reasonaing")
    sum_of_prime_numbers = dspy.OutputField(desc="no need for reasonaing")
    sum_is_even_or_odd = dspy.OutputField(desc="no need for reasonaing")
    reasoning = dspy.OutputField(desc="Give a step by step reasoning")

class WordMathProblem(dspy.Signature):
    """Given a word math problem, solve the problem and provide step by step
       explanation as the output. Do make an answer or round the answer. 
    """
    problem = dspy.InputField()
    explanation = dspy.OutputField()

class ChainOfThoughtSignature(dspy.Signature):
    """
    Given an input text, solve the problem.
    Think through this step by step. Solve each step,
    output the result, and explain how you arrived at your answer
    """
    
    problem_text = dspy.InputField()
    result = dspy.OutputField(desc="no need for reasoning")
    reasoning = dspy.OutputField(desc="Give a step by step reasoning")

class COT(dspy.Module):
    """Chain of Thought Module"""
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought(ChainOfThoughtSignature, max_iters=5)

    def forward(self, problem_text: str):
        return self.cot(problem_text=problem_text)
    
class ProgramOfThoughtSignature(dspy.Signature):
    """
    Given a question, solve the problem.
    """
    
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Give a step by step reasoning and generate Python code")

class POT(dspy.Module):
    """Program of Thought Module"""
    def __init__(self):
        super().__init__()
        self.pot = dspy.ProgramOfThought(ProgramOfThoughtSignature, max_iters=5)

    def forward(self, question: str):
        return self.pot(question=question)
    
class RAGSignature(dspy.Signature):
    """
    Given a context, question, answer the question.
    """
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()
    
class RAG(dspy.Module) :
    def __init__ ( self , num_passages=3) :
        super().__init__()
        # Retrieve will use the user’s default retrieval settings unless overriden .
        self.retrieve = dspy.Retrieve(k=num_passages)
        # ChainOfThought with signature that generates answers given retrieval context & question .
        self.generate_answer = dspy.ChainOfThought (RAGSignature)

    def forward (self, question) :
        context = self.retrieve (question).passages
        return self.generate_answer(context=context, question=question)

class ThoughtReflection (dspy.Module ) :
    def __init__ ( self, num_attempts=5) :
        self.predict = dspy.ChainOfThought (QuestionAnswer, n=num_attempts)
        self.compare = dspy.MultiChainComparison(QuestionAnswer, M=num_attempts)
    
    def forward (self,  question) :
        completions = self.predict(question=question).completions
        return self.compare(question=question, completions=completions)
    