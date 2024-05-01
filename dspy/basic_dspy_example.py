import warnings
import os
from dotenv import load_dotenv, find_dotenv
import openai
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

# Setup environment

ollama_mistral = dspy.OllamaLocal(model='mistral')
dspy.settings.configure(lm=ollama_mistral)

# Load dataset
gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]

# view the first example
print(gsm8k_trainset[:1])

# Define a Module fo solving the GSM8K task
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)
    
# Compile and Evaluate the model on the GSM8K dataset
from dspy.teleprompt import BootstrapFewShot

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

# Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset, valset=gsm8k_devset)

# View the optimized model
from dspy.evaluate import Evaluate

# Set up the evaluator, which can be used multiple times.
evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)

# Evaluate our `optimized_cot` program.
evaluate(optimized_cot)

# Inspect the history of the optimization

ollama_mistral.inspect_history(n=1)




