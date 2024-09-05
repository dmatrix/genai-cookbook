#
# Borrowed and modified from: https://dspy-docs.vercel.app/docs/tutorials/simplified-baleen
# This example demonstrates how to use an unoptimized pipeline in DSPy.
# The example uses the Llama3 model, served by OLlama, to perform the tasks.

import argparse

import dspy
import warnings
from dspy.datasets import HotPotQA
from dspy.evaluate.evaluate import Evaluate
from dspy_utils import gold_passages_retrieved, BOLD_BEGIN, BOLD_END, SimplifiedPipeline, downlad_dataset, parse_args

# Questions to ask the RAG program using the HotPotQA dataset
# later we we'll use the questions to test the unoptimized pipeline
QUESTIONS = [
        "Which  American actor was Candace Kita guest starred with",
        "At My Window was released by which American singer-songwriter?",
        "Who conducts the draft in which Marc-Andre Fleury was drafted to the Vegas Golden Knights for the 2017-18 season?"
]            

from dspy_utils import BOLD_BEGIN, BOLD_END, SimplifiedPipeline

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Parse the command line arguments
    args = parse_args()
    devset_size = args.devset
    trainset_size = args.trainset
    debug = args.debug

    # Setup OLlama environment on the local machine
    ollama_llama3 = dspy.OllamaLocal(model='llama3',
                                      max_tokens=3000)
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
    dspy.settings.configure(lm=ollama_llama3, rm=colbertv2_wiki17_abstracts)

    # Print using the model Llama3
    print(f"{BOLD_BEGIN}Using the {ollama_llama3.model_name} model{BOLD_END}")
    print(f"{BOLD_BEGIN}Using the ColBERTv2 at (url='http://20.102.90.50:2017/wiki17_abstracts) for retrieval{BOLD_END}")
    print(f"{BOLD_BEGIN}Using the SimplifiedPipeline as unoptimized pipeline{BOLD_END}")
    print("--------------------------")

    # Create our Zero Short unoptimized pipeline Signature instance
    # Execute our simplified and unoptimized pipeline
    # Ask any question you like to this simple RAG program.
    
    for question in QUESTIONS:
        # Get the prediction. This contains `pred.context` and `pred.answer`.
        uncompiled_pipeline = SimplifiedPipeline()  # uncompiled (i.e., zero-shot) program
        pred = uncompiled_pipeline(question)

        # Print the contexts and the answer.
        print(f"Question: {question}")
        print(f"Contexts: {pred.context}")
        print(f"Predicted Answer: {pred.answer}")
        print("--------------------------") 
   
    # Inspect the prompt history 
    if debug:
        print(f"{BOLD_BEGIN} Prompt History {BOLD_END}:")
        print(ollama_llama3.inspect_history(n=3))
        print("--------------------------")

    trainset, devset= downlad_dataset(trainset_size=trainset_size,
                                      devset_size=devset_size,
                                      debug=debug)
    if debug:
        print(f"size of trainset:{len(trainset)}; devset:{len(devset)}")

    print(f"{BOLD_BEGIN}Evaluating the unoptimized pipeline ....{BOLD_END}")
    # Set up the `evaluate_on_hotpotqa` function. 
    evaluate_on_hotpotqa = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)
    # Evaluate the uncompiled pipeline on the HotPotQA dataset
    uncompiled_retrieval_score = evaluate_on_hotpotqa(uncompiled_pipeline, metric=gold_passages_retrieved) 

    print(f"## Retrieval Score for uncompiled pipeline: {uncompiled_retrieval_score}")
    print("--------------------------")