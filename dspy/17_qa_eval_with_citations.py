import argparse
import warnings
import random
from tqdm import tqdm

import dspy

from dspy_utils import BOLD_BEGIN, BOLD_END, downlad_dataset, LongFormQA, extract_cited_titles_from_paragraph, extract_cited_titles_from_contexts
from dspy_utils import answer_correctness, citation_faithfulness, calculate_recall, calculate_precision
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

def get_cmdline_args(debug=False, trainset_size=300, devset_size=300):
    # Create the parser
    parser = argparse.ArgumentParser(description='Parse command line arguments.')

    # Add the arguments with default values
    parser.add_argument('--debug', type=bool, default=debug, help='Debug mode (default: {debug})')
    parser.add_argument('--trainset', type=int, default=trainset_size, help='Training set size (default: {trainset_size})')
    parser.add_argument('--devset', type=int, default=devset_size, help='Development set size (default: {devset_size})')
    parser.add_argument('--num_questions', type=int, default=3, help='Number of questions to ask (default: 3)')

    # Parse the arguments
    args = parser.parse_args()
    return args

def evaluate(module,  devset_size=300, debug=False):
    correctness_values = []
    recall_values = []
    precision_values = []
    citation_faithfulness_values = []
    for i, _ in enumerate(tqdm(range(devset_size), desc="Evaluating the model on the devset for correctness and faithfulness")):
        example = devset[i]
        try:
            pred = module(question=example.question)
            correctness_values.append(answer_correctness(example, pred))
            citation_faithfulness_score, _ = citation_faithfulness(None, pred, None)
            citation_faithfulness_values.append(citation_faithfulness_score)
            recall = calculate_recall(example, pred)
            precision = calculate_precision(example, pred)
            recall_values.append(recall)
            precision_values.append(precision)
            if debug:
                print(f"Correctness-{i+1}: {correctness_values[-1]}")
                print(f"Recall-{i+1}: {recall_values[-1]}")
                print(f"Precision{i+1}: {precision_values[-1]}")
                print(f"Citation Faithfulness{i+1}: {citation_faithfulness_values[-1]}")
        except Exception as e:
            print(f"Failed generation with error: {e}")

    average_correctness = sum(correctness_values) / devset_size if correctness_values else 0
    average_recall = sum(recall_values) / devset_size if recall_values else 0
    average_precision = sum(precision_values) / devset_size if precision_values else 0
    average_citation_faithfulness = sum(citation_faithfulness_values) / devset_size if citation_faithfulness_values else 0

    print("--------------------------")
    print(f"Final results for the model on the devset of size {devset_size}:")
    print(f"Average Correctness: {average_correctness}")
    print(f"Average Recall: {average_recall}")
    print(f"Average Precision: {average_precision}")
    print(f"Average Citation Faithfulness: {average_citation_faithfulness}")
    print("--------------------------")



if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    # Parse the command line arguments
    args = get_cmdline_args()
    print(args)

    devset_size = args.devset
    trainset_size = args.trainset
    debug = args.debug
    num_questions = args.num_questions

    print(f"{BOLD_BEGIN}Command line arguments: {args} {BOLD_END}")

    # Setup OLlama environment on the local machine
    ollama_llama3 = dspy.OllamaLocal(model='llama3',
                                      max_tokens=3000)
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
    dspy.settings.configure(lm=ollama_llama3, rm=colbertv2_wiki17_abstracts)

    # Print using the model Llama3
    print(f"{BOLD_BEGIN}Using the {ollama_llama3.model_name} model{BOLD_END}")
    print(f"{BOLD_BEGIN}Using the ColBERTv2 at (url='http://20.102.90.50:2017/wiki17_abstracts) for retrieval{BOLD_END}")
    print(f"{BOLD_BEGIN}Downloading HotPotQA dataset...{BOLD_END}")
    
    # Download the dataset
    trainset, devset= downlad_dataset(trainset_size=trainset_size,
                                       devset_size=devset_size,
                                       debug=debug)
    print(f"{BOLD_BEGIN}Using the LongformerQA model{BOLD_END}")
    print("--------------------------")

    long_form_qa = LongFormQA()
    for idx, number in enumerate(tqdm(range(num_questions), desc="Processing Questions from devset for Evaluation")):
        rand_idx = random.randint(0, len(devset))
        example = devset[rand_idx]
        question = devset[rand_idx].question
        gold_titles = devset[rand_idx].gold_titles
        print(f"{BOLD_BEGIN}Question-{idx+1}{BOLD_END}: {question}")
        print(f"{BOLD_BEGIN}Relevant Wikipedia Titles for Question-{idx+1}: {BOLD_END}{gold_titles}")
        pred = long_form_qa(question)
        print(f"{BOLD_BEGIN}Answer-{idx+1} context{BOLD_END}: {pred.context}")
        context_titles = extract_cited_titles_from_contexts(pred.context)
        print(f"{BOLD_BEGIN}Answer-{idx+1} titles in context {BOLD_END}: {context_titles}")
        print(f"{BOLD_BEGIN}Answer-{idx+1} citations paragraphs{BOLD_END}: {pred.paragraph}")
        cited_articles = extract_cited_titles_from_paragraph(pred.paragraph, pred.context)
        print(f"{BOLD_BEGIN}Answer-{idx+1} cited articles{BOLD_END}: {cited_articles}")
        print("--------------------------")
        print(f"{BOLD_BEGIN}Returned full response Answer-{idx+1}{BOLD_END}: {pred}")
        print("--------------------------")
        print("\n")
        citation_faithfulness_score, _ = citation_faithfulness(None, pred, None)
        print(f"{BOLD_BEGIN}Predicted Paragraph:{BOLD_END} {pred.paragraph}")
        print(f"{BOLD_BEGIN}Citation Faithfulness: {BOLD_END} {citation_faithfulness_score}")
        print("-----------------------")
        print("\n")
    # Let try to evaluate for correctness and faithfulness
    devset_size = 100
    print(f"{BOLD_BEGIN}Evaluating the model on the dev set for {devset_size} samples{BOLD_END}...")
    evaluate(long_form_qa, devset_size=100, debug=True)
    print("--------------------------")
