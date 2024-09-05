import argparse
import re

import dspy
from typing import List
from dspy.datasets import HotPotQA
from dsp.utils import EM, normalize_text


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

class ZeroShotEntityNameRecognition(dspy.Signature):
    """Given a text, identify the named entities in the text. 
       Output the named entities as persons,organizations, 
       and locations.
    """

    text = dspy.InputField()
    entities = dspy.OutputField()


class DialogueGeneration(dspy.Signature):
    """Generate a dialogue between a customer and Agent based
      on the problem input text of a technical problem. Provide the dialogue as the output.
    """

    problem_text = dspy.InputField()
    dialogue = dspy.OutputField(prefix="Dialogue between customer and support agent:")


    
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

## New Signatures for optimized pipeline

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

#
# Given and input question and context, generate the response
# as a paragraph with citations from the retreived context from
# the retreiver
class GenerateCitedParagraph(dspy.Signature):
    """Generate a paragraph with citations."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    paragraph = dspy.OutputField(desc="includes citations")

# Build the optimized pipeline
# Comprises a collection of serial modules that are executed in sequence
# generate_query (GenerateSearchQuery) -> retrieve (Retrieve) -> generate_answer(GenerateAnswer)
from dsp.utils import deduplicate

class SimplifiedPipeline(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2, debug=False):
        super().__init__()

        # generate a query for each hop
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        # retrieve k passages for each hop
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        # generate an answer
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
        self.debug = debug
    
    def forward(self, question):
        """Answer a question by generating a query, retrieving passages, and generating an answer."""
        context = []
        # Control flow loop for the pipeline
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            if self.debug:
                print(f"Query for hop {hop + 1}: {query}")
                print(f"context: {context}...")
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
            if self.debug:
                print(f"Retrieved Contexts: {[c + '<eoc>' for c in context]}")
                print(f"Total context length: {len(context)}")
                

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)
    
# Define metric to check if we retrieved the correct documents
def gold_passages_retrieved(example, pred, trace=None):
    gold_titles = set(map(dspy.evaluate.normalize_text, example["gold_titles"]))
    found_titles = set(
        map(dspy.evaluate.normalize_text, [c.split(" | ")[0] for c in pred.context])
    )
    return gold_titles.issubset(found_titles)

# Define metric to check if we retrieved the correct documents
# Let's first define our validation logic for compilation:
# 1. The predicted answer matches the gold answer.
# 2. The retrieved context contains the gold answer.
# 3. None of the generated queries is rambling (i.e., none exceeds 100 characters in length).
# 4. None of the generated queries is roughly repeated (i.e., none is within 0.8 or higher F1 score of earlier queries).
def validate_context_and_answer_and_hops(example, pred, trace=None):
    if not dspy.evaluate.answer_exact_match(example, pred): return False
    if not dspy.evaluate.answer_passage_match(example, pred): return False

    # check if the question appears in the output, suggesting that the pipeline is further refining the question
    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    return True

# Download the HotPotQA dataset
def downlad_dataset(trainset_size=25, devset_size=50, debug=False):
    # Load the dataset.
    dataset = HotPotQA(train_seed=1, train_size=trainset_size, eval_seed=2023, dev_size=devset_size, test_size=0)

    # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
    trainset = [x.with_inputs('question') for x in dataset.train]
    devset = [x.with_inputs('question') for x in dataset.dev]
    if debug:
        print(f"trainset[:3]:{trainset[:3]}")
        print(f"devset[:3]:{devset[:3]}")

    return trainset, devset

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Parse command line arguments.')

    # Add the arguments with default values
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode (default: False)')
    parser.add_argument('--devset', type=int, default=50, help='Development set size (default: 50)')
    parser.add_argument('--trainset', type=int, default=20, help='Training set size (default: 25)')
    parser.add_argument('--num_threads', type=int, default=2, help='Number of threads for evaluation (default: 2)')


    # Parse the arguments
    args = parser.parse_args()
    return args

def extract_cited_titles_from_paragraph(paragraph, context):
    cited_indices = [int(m.group(1)) for m in re.finditer(r'\[(\d+)\][\.,]?', paragraph)]
    cited_indices = [index - 1 for index in cited_indices if index <= len(context)]
    cited_titles = [context[index].split(' | ')[0] for index in cited_indices]
    return cited_titles

def extract_cited_titles_from_contexts(context):
    return [c.split(' | ')[0] for c in context]

def extract_text_by_citation(paragraph):
    citation_regex = re.compile(r'(.*?)(\[\d+\][\.,]?)', re.DOTALL)
    parts_with_citation = citation_regex.findall(paragraph)
    citation_dict = {}
    for part, citation in parts_with_citation:
        part = part.strip()
        citation_num = re.search(r'\[(\d+)\][\.,]?', citation).group(1)
        citation_dict.setdefault(str(int(citation_num) - 1), []).append(part)
    return citation_dict

def calculate_recall(example, pred, trace=None):
    gold_titles = set(example['gold_titles'])
    found_cited_titles = set(extract_cited_titles_from_paragraph(pred.paragraph, pred.context))
    intersection = gold_titles.intersection(found_cited_titles)
    recall = len(intersection) / len(gold_titles) if gold_titles else 0
    return recall

def calculate_precision(example, pred, trace=None):
    gold_titles = set(example['gold_titles'])
    found_cited_titles = set(extract_cited_titles_from_paragraph(pred.paragraph, pred.context))
    intersection = gold_titles.intersection(found_cited_titles)
    precision = len(intersection) / len(found_cited_titles) if found_cited_titles else 0
    return precision

def answer_correctness(example, pred, trace=None):
    assert hasattr(example, 'answer'), "Example does not have 'answer'."
    normalized_context = normalize_text(pred.paragraph)
    if isinstance(example.answer, str):
        gold_answers = [example.answer]
    elif isinstance(example.answer, list):
        gold_answers = example.answer
    else:
        raise ValueError("'example.answer' is not string or list.")
    return 1 if any(normalize_text(answer) in normalized_context for answer in gold_answers) else 0

# The main module that constructs a response from the question and context
# with citations.
# Anan

class LongFormQA(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()
        # Generate query for each hop
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        # Retrieve k messages 
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        # Use signature for generating cited messages
        self.generate_cited_paragraph = dspy.ChainOfThought(GenerateCitedParagraph)
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        # Iterate over number of hops

        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        pred = self.generate_cited_paragraph(context=context, question=question)
        pred = dspy.Prediction(context=context, paragraph=pred.paragraph)
        return pred

#
# Define the signature for the CheckCitationFaithfulness module
# This module will verify that the text is based on the provided context.
#
class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""
    context = dspy.InputField(desc="may contain relevant facts")
    text = dspy.InputField(desc="between 1 to 2 sentences")
    faithfulness = dspy.OutputField(desc="boolean indicating if text is faithful to context")

def citation_faithfulness(example, pred, trace):
    paragraph, context = pred.paragraph, pred.context
    citation_dict = extract_text_by_citation(paragraph)
    if not citation_dict:
        return False, None
    context_dict = {str(i): context[i].split(' | ')[1] for i in range(len(context))}
    faithfulness_results = []
    unfaithful_citations = []
    check_citation_faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
    for citation_num, texts in citation_dict.items():
        if citation_num not in context_dict:
            continue
        current_context = context_dict[citation_num]
        for text in texts:
            try:
                result = check_citation_faithfulness(context=current_context, text=text)
                is_faithful = result.faithfulness.lower() == 'true'
                faithfulness_results.append(is_faithful)
                if not is_faithful:
                    unfaithful_citations.append({'paragraph': paragraph, 'text': text, 'context': current_context})
            except ValueError as e:
                faithfulness_results.append(False)
                unfaithful_citations.append({'paragraph': paragraph, 'text': text, 'error': str(e)})
    final_faithfulness = all(faithfulness_results)
    if not faithfulness_results:
        return False, None
    return final_faithfulness, unfaithful_citations

