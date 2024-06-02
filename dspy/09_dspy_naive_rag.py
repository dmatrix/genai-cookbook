import dspy
import warnings
from dspy_utils import RAG, BOLD_BEGIN, BOLD_END

# Questions to ask the RAG model
QUESTIONS = [
    "What is the capital of Tanzania?",
    "Who was the president of the United States in 1960?",
    "What is the largest mammal?",
    "What is the most populous country?",
    "What is the most widely spoken language?",
    "Which country won the FIFA Football World Cup in 1970?",
    "Which country has won the most FIFA Football World Cups?",
    "Who is the author of the book '1984'?",
    "What is the most popular programming language?",
    "When was the last Solar Eclipse in the United States, and what states were covered in total darkness?"
]
    
if __name__ == "__main__":

    # Filter out warnings
    warnings.filterwarnings("ignore")

    # Instantiate our Language Model
    # Setup OLlama environment on the local machine
    ollama_mistral = dspy.OllamaLocal(model='mistral',
                                      max_tokens=2500)
    # Instantiate the ColBERTv2 as Retrieval module
    colbert_rm = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

    # Configure the settings
    dspy.settings.configure(lm=ollama_mistral, rm=colbert_rm)

    # Instantiate the RAG module
    rag = RAG(num_passages=5)
    for idx, question in enumerate(QUESTIONS):
        print(f"{BOLD_BEGIN}Question {idx + 1}: {BOLD_END}{question}")
        response = rag(question=question)
        print(f"{BOLD_BEGIN}Answer    : {BOLD_END}{response.answer}")
        print("-----------------------------\n")