#
# Borrowed and mildly modified from: https://dspy-docs.vercel.app/docs/tutorials/simplified-baleen
# This example demonstrates how to use an unoptimized pipeline in DSPy.
# The example uses the Llama3 model, served by OLlama, to perform the tasks.

import dspy
import warnings
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch
from dspy_utils import gold_passages_retrieved, BOLD_BEGIN, BOLD_END, SimplifiedPipeline, downlad_dataset, validate_context_and_answer_and_hops        
from dspy_utils import BOLD_BEGIN, BOLD_END, SimplifiedPipeline, parse_args



if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Parse the command line arguments
    args = parse_args()
    devset_size = args.devset
    trainset_size = args.trainset
    debug = args.debug
    num_threads = args.num_threads

    # print the command line arguments
    print(f"{BOLD_BEGIN}Command line arguments: {args} {BOLD_END}")

    # Setup OLlama environment on the local machine
    ollama_llama3 = dspy.OllamaLocal(model='llama3',
                                      max_tokens=3000)
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
    dspy.settings.configure(lm=ollama_llama3, rm=colbertv2_wiki17_abstracts)

    # Print using the model Llama3
    print(f"{BOLD_BEGIN}Using the {ollama_llama3.model_name} model{BOLD_END}")
    print(f"{BOLD_BEGIN}Using the ColBERTv2 at (url='http://20.102.90.50:2017/wiki17_abstracts) for retrieval{BOLD_END}")
    print(f"{BOLD_BEGIN}Using the SimplifiedPipeline as compiled and optimized pipeline{BOLD_END}")
    print("--------------------------")

    # Download the dataset
    trainset, devset= downlad_dataset(trainset_size=trainset_size,
                                      devset_size=devset_size,
                                      debug=debug)
    if debug:
        print(f"size of trainset:{len(trainset)}; devset:{len(devset)}")


    # Create and optimzer and compile the pipeline and optimize it
    optimzer = BootstrapFewShot(metric=validate_context_and_answer_and_hops)

    print(f"{BOLD_BEGIN}Compiling the pipeline ....{BOLD_END}")
    compiled_pipeline = optimzer.compile(SimplifiedPipeline(), teacher=SimplifiedPipeline(passages_per_hop=2), trainset=trainset)    

    print(f"{BOLD_BEGIN}Evaluating the compiled and optimized pipeline ....{BOLD_END}")
    # Set up the `evaluate_on_hotpotqa` function. 
    evaluate_on_hotpotqa = Evaluate(devset=devset, num_threads=num_threads, display_progress=True, display_table=5)

    # Evaluate the compiled pipeline on the HotPotQA dataset
    compiled_pipeline_retrieval_score = evaluate_on_hotpotqa(compiled_pipeline, metric=gold_passages_retrieved)
    print(f"## Retrieval Score for compiled pipeline: {compiled_pipeline_retrieval_score}")
    print("--------------------------")

    # Saving the optimized pipeline
    print(f"{BOLD_BEGIN}Saving the optimized pipeline ....{BOLD_END}")
    compiled_pipeline.save("optimized_pipeline")


